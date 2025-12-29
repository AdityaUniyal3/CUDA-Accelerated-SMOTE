#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <numeric> 
#include <string>

using namespace std;

struct Args {
    std::string in, out = "output_balanced.csv";
    int k = 5, seed = 42;
    bool quiet = false;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--in"    && i+1 < argc) a.in   = argv[++i];
        else if (s == "--out"   && i+1 < argc) a.out  = argv[++i];
        else if (s == "--k"     && i+1 < argc) a.k    = std::stoi(argv[++i]);
        else if (s == "--seed"  && i+1 < argc) a.seed = std::stoi(argv[++i]);
        else if (s == "--quiet")                a.quiet = true;
    }
    return a;
}

// Data structure to hold a single row of data
struct DataPoint {
    vector<double> features;
    int label;  // Assumes binary labels (0 = majority, 1 = minority)
};

// Normalize features to [0,1] range to prevent scale bias
void normalizeFeatures(vector<DataPoint>& data) {
    if (data.empty()) return;
    
    size_t numFeatures = data[0].features.size();
    vector<double> mins(numFeatures, numeric_limits<double>::max());
    vector<double> maxs(numFeatures, numeric_limits<double>::lowest());
    
    // Find min and max for each feature
    for (const auto& point : data) {
        for (size_t i = 0; i < numFeatures; i++) {
            mins[i] = min(mins[i], point.features[i]);
            maxs[i] = max(maxs[i], point.features[i]);
        }
    }
    
    // Apply normalization
    for (auto& point : data) {
        for (size_t i = 0; i < numFeatures; i++) {
            if (maxs[i] - mins[i] > 1e-9) { // Avoid division by zero
                point.features[i] = (point.features[i] - mins[i]) / (maxs[i] - mins[i]);
            }
        }
    }
}

double euclideanDistance(const DataPoint& a, const DataPoint& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.features.size(); i++) {
        sum += pow(a.features[i] - b.features[i], 2);
    }
    return sqrt(sum);
}

// Improved KNN with distance weighting
vector<pair<size_t, double>> findWeightedKNN(const vector<DataPoint>& minorityData, const DataPoint& sample, int k) {
    vector<pair<double, size_t>> distances;
    
    for (size_t i = 0; i < minorityData.size(); i++) {
        // Skip identical samples
        if (euclideanDistance(sample, minorityData[i]) < 1e-9) continue;
        
        double dist = euclideanDistance(sample, minorityData[i]);
        distances.emplace_back(dist, i);
    }
    
    // Partial sort to find top-k neighbors
    partial_sort(distances.begin(), distances.begin() + min(k, (int)distances.size()), distances.end());
    
    // Convert to weighted neighbors (closer neighbors have more weight)
    vector<pair<size_t, double>> weightedNeighbors;
    double totalInverseDistance = 0.0;
    
    for (int i = 0; i < min(k, (int)distances.size()); i++) {
        double invDist = 1.0 / (distances[i].first + 1e-9); // Avoid division by zero
        totalInverseDistance += invDist;
        weightedNeighbors.emplace_back(distances[i].second, invDist);
    }
    
    // Normalize weights
    for (auto& neighbor : weightedNeighbors) {
        neighbor.second /= totalInverseDistance;
    }
    
    return weightedNeighbors;
}

vector<DataPoint> balancedSmote(const vector<DataPoint>& minorityData, const vector<DataPoint>& majorityData, int k = 5) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);
    
    // Calculate needed samples (optionally keep some imbalance if desired)
    int minorityCount = minorityData.size();
    int majorityCount = majorityData.size();
    int needed = majorityCount - minorityCount;
    
    vector<DataPoint> syntheticSamples;
    
    // First identify "danger zone" samples near decision boundary
    vector<DataPoint> dangerSamples;
    for (const auto& sample : minorityData) {
        auto neighbors = findWeightedKNN(minorityData, sample, k);
        int majorityNeighbors = 0;
        
        for (const auto& neighbor : neighbors) {
            if (minorityData[neighbor.first].label == 0) {
                majorityNeighbors++;
            }
        }
        
        // Consider samples with at least one majority neighbor as "in danger"
        if (majorityNeighbors > 0) {
            dangerSamples.push_back(sample);
        }
    }
    
    // If no danger samples found, use all minority samples
    if (dangerSamples.empty()) {
        dangerSamples = minorityData;
    }
    
    // Calculate samples per point, weighted by their "danger" level
    vector<double> weights;
    for (const auto& sample : dangerSamples) {
        auto neighbors = findWeightedKNN(minorityData, sample, k);
        double dangerScore = 0.0;
        for (const auto& neighbor : neighbors) {
            if (minorityData[neighbor.first].label == 0) {
                dangerScore += neighbor.second;
            }
        }
        weights.push_back(dangerScore);
    }
    
    // Normalize weights
    double totalWeight = accumulate(weights.begin(), weights.end(), 0.0);
    if (totalWeight < 1e-9) {
        fill(weights.begin(), weights.end(), 1.0 / weights.size());
    } else {
        for (auto& w : weights) w /= totalWeight;
    }
    
    // Generate synthetic samples
    discrete_distribution<int> dist(weights.begin(), weights.end());
    while (syntheticSamples.size() < needed) {
        // Select a sample based on danger weights
        int sampleIdx = dist(gen);
        const DataPoint& sample = dangerSamples[sampleIdx];
        
        // Get weighted neighbors
        auto neighbors = findWeightedKNN(minorityData, sample, k);
        
        // Select neighbor based on weights
        vector<double> neighborWeights;
        for (const auto& neighbor : neighbors) {
            neighborWeights.push_back(neighbor.second);
        }
        discrete_distribution<int> neighborDist(neighborWeights.begin(), neighborWeights.end());
        int neighborIdx = neighborDist(gen);
        const DataPoint& neighbor = minorityData[neighbors[neighborIdx].first];
        
        // Create synthetic sample with random interpolation
        DataPoint synthetic;
        double gap = dis(gen);
        
        // Occasionally create extrapolated samples
        if (dis(gen) < 0.2) gap = 1.0 + dis(gen) * 0.5; // 1.0-1.5
        else if (dis(gen) < 0.4) gap = -0.5 + dis(gen) * 0.5; // -0.5-0.0
        
        for (size_t j = 0; j < sample.features.size(); j++) {
            synthetic.features.push_back(
                sample.features[j] + gap * (neighbor.features[j] - sample.features[j])
            );
        }
        synthetic.label = 1;
        syntheticSamples.push_back(synthetic);
    }
    
    return syntheticSamples;
}

std::vector<DataPoint> readCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Failed to open file: " + filename);

    std::vector<DataPoint> dataset;
    std::string line;
    bool first = true;

    auto looks_like_header = [](const std::string& s){
        for (unsigned char c : s) if (std::isalpha(c)) return true;
        return false;
    };

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (first && looks_like_header(line)) { first = false; continue; }
        first = false;

        std::stringstream ss(line);
        std::vector<double> vals; vals.reserve(64);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty()) vals.push_back(std::stod(tok));
        }
        if (vals.empty()) continue;

        DataPoint p;
        p.label = static_cast<int>(vals.back());
        vals.pop_back();
        p.features.assign(vals.begin(), vals.end());
        dataset.push_back(std::move(p));
    }
    return dataset;
}


// Write DataPoints to a CSV file
void writeCSV(const string& filename, const vector<DataPoint>& data) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }

    for (const auto& point : data) {
        for (size_t i = 0; i < point.features.size(); i++) {
            file << point.features[i];
            if (i < point.features.size() - 1) file << ",";
        }
        file << "," << point.label << "\n";
    }
}

int main(int argc, char** argv) {
    try {
        Args args = parseArgs(argc, argv);
        if (args.in.empty()) { std::cerr << "Usage: --in <file.csv> [--out <file.csv>] [--k <int>] [--seed <int>] [--quiet]\n"; return 2; }

        std::vector<DataPoint> data = readCSV(args.in);              // <— use args.in
        std::vector<DataPoint> minority, majority;
        for (const auto& p : data) (p.label == 1 ? minority : majority).push_back(p);

        if (!args.quiet) {
            std::cout << "Original distribution:\n";
            std::cout << "  Majority class (0): " << majority.size() << " samples\n";
            std::cout << "  Minority class (1): " << minority.size() << " samples\n";
        }

        // If balancedSmote seeds its own RNG, change it to accept a seed (or set a global mt19937 with args.seed)
        auto synth = balancedSmote(minority, majority, args.k /*, args.seed */);

        std::vector<DataPoint> balanced = majority;
        balanced.insert(balanced.end(), minority.begin(), minority.end());
        balanced.insert(balanced.end(), synth.begin(), synth.end());

        if (!args.quiet) {
            std::cout << "\nBalanced distribution:\n";
            std::cout << "  Majority class (0): " << majority.size() << " samples\n";
            std::cout << "  Minority class (1): " << minority.size() + synth.size() << " samples\n";
        }

        writeCSV(args.out, balanced);                                // <— use args.out
        if (!args.quiet) std::cout << "\nBalanced data saved to: " << args.out << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
