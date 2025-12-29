#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <chrono>

using namespace std;

#define CUDA_CHECK(err) \
    do { \
        if (err != cudaSuccess) { \
            cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(1); \
        } \
    } while (0)

// Data structure for GPU
struct GPUPointer {
    double* features;
    int label;
};

// Data structure for CPU
struct DataPoint {
    vector<double> features;
    int label;
};

// Kernel for parallel distance calculation
__global__ void computeDistancesKernel(const double* features, const double* query, double* distances, int numFeatures, int numSamples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSamples) {
        double sum = 0.0;
        for (int i = 0; i < numFeatures; i++) {
            double diff = features[idx * numFeatures + i] - query[i];
            sum += diff * diff;
        }
        distances[idx] = sqrt(sum);
    }
}


// Kernel for synthetic sample generation
__global__ void generateSamplesKernel(const double* samples, const double* neighbors, double* synthetic, const int* pairs, const float* gaps, int numFeatures, int numSynthetic) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numSynthetic) {
        int sampleIdx = pairs[idx * 2];
        int neighborIdx = pairs[idx * 2 + 1];
        float gap = gaps[idx];
        
        for (int i = 0; i < numFeatures; i++) {
            synthetic[idx * numFeatures + i] = 
                samples[sampleIdx * numFeatures + i] + 
                gap * (samples[neighborIdx * numFeatures + i] - 
                      samples[sampleIdx * numFeatures + i]);
        }
    }
}



// Helper function to transfer data to GPU
void copyDataToGPU(const vector<DataPoint>& data, double*& d_features, int*& d_labels, int numFeatures) {
    // Flatten features
    vector<double> flatFeatures;
    vector<int> labels;
    for (const auto& point : data) {
        flatFeatures.insert(flatFeatures.end(), point.features.begin(), point.features.end());
        labels.push_back(point.label);
    }
    
    // Allocate and copy to GPU
    CUDA_CHECK(cudaMalloc(&d_features, flatFeatures.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_labels, labels.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_features, flatFeatures.data(), flatFeatures.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, labels.data(), labels.size() * sizeof(int), cudaMemcpyHostToDevice));
}



// GPU-accelerated KNN with distance weighting
vector<pair<size_t, double>> findWeightedKNN_GPU(const vector<DataPoint>& minorityData, const DataPoint& sample, int k) {
    // Prepare data for GPU
    int numFeatures = sample.features.size();
    int numSamples = minorityData.size();
    
    // Copy minority data to GPU
    double* d_minorityFeatures;
    int* d_minorityLabels;
    copyDataToGPU(minorityData, d_minorityFeatures, d_minorityLabels, numFeatures);
    
    // Copy query sample to GPU
    double* d_query;
    CUDA_CHECK(cudaMalloc(&d_query, numFeatures * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_query, sample.features.data(), numFeatures * sizeof(double), cudaMemcpyHostToDevice));
    
    // Allocate distances array on GPU
    double* d_distances;
    CUDA_CHECK(cudaMalloc(&d_distances, numSamples * sizeof(double)));
    
    // Launch distance calculation kernel
    int blockSize = 256;
    int numBlocks = (numSamples + blockSize - 1) / blockSize;
    computeDistancesKernel<<<numBlocks, blockSize>>>(d_minorityFeatures, d_query, d_distances, numFeatures, numSamples);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy distances back to CPU
    vector<double> distances(numSamples);
    CUDA_CHECK(cudaMemcpy(distances.data(), d_distances, numSamples * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Find top-k neighbors on CPU (or use thrust::sort on GPU for better performance)
    vector<pair<double, size_t>> distancePairs;
    for (size_t i = 0; i < distances.size(); i++) {
        if (distances[i] < 1e-9) continue; // Skip identical samples
        distancePairs.emplace_back(distances[i], i);
    }
    
    partial_sort(distancePairs.begin(), distancePairs.begin() + min(k, (int)distancePairs.size()), distancePairs.end());
    
    // Convert to weighted neighbors
    vector<pair<size_t, double>> weightedNeighbors;
    double totalInverseDistance = 0.0;
    
    for (int i = 0; i < min(k, (int)distancePairs.size()); i++) {
        double invDist = 1.0 / (distancePairs[i].first + 1e-9);
        totalInverseDistance += invDist;
        weightedNeighbors.emplace_back(distancePairs[i].second, invDist);
    }
    
    // Normalize weights
    for (auto& neighbor : weightedNeighbors) {
        neighbor.second /= totalInverseDistance;
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_minorityFeatures));
    CUDA_CHECK(cudaFree(d_minorityLabels));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_distances));
    
    return weightedNeighbors;
}



// GPU-accelerated synthetic sample generation
vector<DataPoint> generateSyntheticGPU(const vector<DataPoint>& dangerSamples, const vector<DataPoint>& minorityData, int needed, int k, mt19937& gen, uniform_real_distribution<double>& dis) {
    int numFeatures = minorityData[0].features.size();
    
    // Prepare data for GPU
    double* d_minorityFeatures;
    int* d_minorityLabels;
    copyDataToGPU(minorityData, d_minorityFeatures, d_minorityLabels, numFeatures);
    
    // Generate sample-neighbor pairs and gaps on CPU
    vector<int> pairs;
    vector<float> gaps;
    
    for (int i = 0; i < needed; i++) {
        // Select a random danger sample
        int sampleIdx = rand() % dangerSamples.size();
        const DataPoint& sample = dangerSamples[sampleIdx];
        
        // Find its neighbors
        auto neighbors = findWeightedKNN_GPU(minorityData, sample, k);
        
        // Select neighbor based on weights
        vector<double> weights;
        for (const auto& neighbor : neighbors) {
            weights.push_back(neighbor.second);
        }
        discrete_distribution<int> neighborDist(weights.begin(), weights.end());
        int neighborIdx = neighborDist(gen);
        
        // Generate gap with occasional extrapolation
        float gap;
        if (dis(gen) < 0.2) gap = 1.0f + dis(gen) * 0.5f;
        else if (dis(gen) < 0.4) gap = -0.5f + dis(gen) * 0.5f;
        else gap = dis(gen);
        
        pairs.push_back(sampleIdx);
        pairs.push_back(neighbors[neighborIdx].first);
        gaps.push_back(gap);
    }
    
    // Copy pairs and gaps to GPU
    int* d_pairs;
    float* d_gaps;
    CUDA_CHECK(cudaMalloc(&d_pairs, pairs.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gaps, gaps.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pairs, pairs.data(), pairs.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gaps, gaps.data(), gaps.size() * sizeof(float), cudaMemcpyHostToDevice));
    
    // Allocate space for synthetic samples on GPU
    double* d_synthetic;
    CUDA_CHECK(cudaMalloc(&d_synthetic, needed * numFeatures * sizeof(double)));
    
    // Launch generation kernel
    int blockSize = 256;
    int numBlocks = (needed + blockSize - 1) / blockSize;
    generateSamplesKernel<<<numBlocks, blockSize>>>(d_minorityFeatures, d_minorityFeatures, d_synthetic, d_pairs, d_gaps, numFeatures, needed);
    CUDA_CHECK(cudaGetLastError());
    
    // Copy synthetic samples back to CPU
    vector<double> syntheticFeatures(needed * numFeatures);
    CUDA_CHECK(cudaMemcpy(syntheticFeatures.data(), d_synthetic, needed * numFeatures * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Convert to DataPoint format
    vector<DataPoint> syntheticSamples;
    for (int i = 0; i < needed; i++) {
        DataPoint point;
        point.features.assign(syntheticFeatures.begin() + i * numFeatures, syntheticFeatures.begin() + (i + 1) * numFeatures);
        point.label = 1;
        syntheticSamples.push_back(point);
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_minorityFeatures));
    CUDA_CHECK(cudaFree(d_minorityLabels));
    CUDA_CHECK(cudaFree(d_pairs));
    CUDA_CHECK(cudaFree(d_gaps));
    CUDA_CHECK(cudaFree(d_synthetic));
    
    return syntheticSamples;
}



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

vector<DataPoint> balancedSmote_GPU(const vector<DataPoint>& minorityData, const vector<DataPoint>& majorityData, int k = 5) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0.0, 1.0);
    
    int minorityCount = minorityData.size();
    int majorityCount = majorityData.size();
    int needed = majorityCount - minorityCount;
    
    // Identify danger samples on CPU (less parallelizable)
    vector<DataPoint> dangerSamples;
    for (const auto& sample : minorityData) {
        auto neighbors = findWeightedKNN_GPU(minorityData, sample, k);
        int majorityNeighbors = 0;
        for (const auto& neighbor : neighbors) {
            if (minorityData[neighbor.first].label == 0) {
                majorityNeighbors++;
            }
        }
        if (majorityNeighbors > 0) {
            dangerSamples.push_back(sample);
        }
    }
    
    if (dangerSamples.empty()) {
        dangerSamples = minorityData;
    }
    
    // Generate synthetic samples on GPU
    return generateSyntheticGPU(dangerSamples, minorityData, needed, k, gen, dis);
}


// Read CSV file into a vector of DataPoints
vector<DataPoint> readCSV(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file: " + filename);
    }

    vector<DataPoint> dataset;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        DataPoint point;
        string value;
        
        // Read features (assume all columns except last are features)
        while (getline(ss, value, ',')) {
            if (ss.peek() == ',') ss.ignore();
            point.features.push_back(stod(value));
        }
        
        // Last column is the label
        point.label = static_cast<int>(point.features.back());
        point.features.pop_back();
        dataset.push_back(point);
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
/* 
----------------------------------------UNIT TESTING-------------------------------------------------
*/
void testDistanceKernel() {
    // Create test data
    vector<DataPoint> testData = {
        {{1.0, 2.0, 3.0}, 0},
        {{4.0, 5.0, 6.0}, 1},
        {{7.0, 8.0, 9.0}, 0}
    };
    DataPoint query = {{1.0, 2.0, 3.0}, 0};
    
    // Run GPU version
    auto gpuResults = findWeightedKNN_GPU(testData, query, 2);
    
    // Run CPU version for comparison
    auto cpuResults = findWeightedKNN(testData, query, 2);
    
    // Verify results
    assert(gpuResults.size() == cpuResults.size());
    for (size_t i = 0; i < gpuResults.size(); i++) {
        assert(abs(gpuResults[i].second - cpuResults[i].second) < 1e-6);
    }
    cout << "Distance kernel test passed!\n";
}

void testSampleGeneration() {
    vector<DataPoint> minority = {
        {{1.0, 2.0}, 1},
        {{3.0, 4.0}, 1}
    };
    vector<DataPoint> majority = {
        {{0.0, 0.0}, 0},
        {{0.0, 0.0}, 0},
        {{0.0, 0.0}, 0}
    };
    
    auto synthetic = balancedSmote_GPU(minority, majority, 2);
    
    assert(synthetic.size() == 1); // 3 majority - 2 minority = 1 needed
    cout << "Sample generation test passed!\n";
}

void goldenTest() {
    vector<DataPoint> minority = {
        {{1.0, 2.0}, 1},
        {{3.0, 4.0}, 1}
    };
    vector<DataPoint> majority = {
        {{0.0, 0.0}, 0},
        {{0.0, 0.0}, 0},
        {{0.0, 0.0}, 0}
    };
    
    auto balanced = balancedSmote_GPU(minority, majority, 2);
    
    // Verify synthetic sample is between minority and its neighbors
    assert(balanced[0].features[0] > 2.0 && balanced[0].features[0] < 10.0);
    cout << "Golden test passed!\n";
}

void benchmark() {
    // Generate synthetic dataset
    vector<DataPoint> bigData;
    for (int i = 0; i < 10000; i++) {
        bigData.push_back({{(double)rand()/RAND_MAX, (double)rand()/RAND_MAX}, i % 10 == 0 ? 1 : 0});
    }
    
    // Separate classes
    vector<DataPoint> minority, majority;
    for (const auto& pt : bigData) {
        (pt.label == 1 ? minority : majority).push_back(pt);
    }
    
    // Time GPU version
    auto start = chrono::high_resolution_clock::now();
    auto gpuResult = balancedSmote_GPU(minority, majority, 5);
    auto gpuTime = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - start).count();
    
    // Time CPU version
    start = chrono::high_resolution_clock::now();
    auto cpuResult = balancedSmote_GPU(minority, majority, 5);
    auto cpuTime = chrono::duration_cast<chrono::milliseconds>(
        chrono::high_resolution_clock::now() - start).count();
    
    cout << "GPU time: " << gpuTime << "ms\n";
    cout << "CPU time: " << cpuTime << "ms\n";
    cout << "Speedup: " << (double)cpuTime/gpuTime << "x\n";
}

void visualizeResults() {
    auto data = readCSV("test_datasets/dataset_5_samples_5000_features_30_imbalance_0.95.csv");
    
    // Separate and process
    vector<DataPoint> minority, majority;
    for (const auto& pt : data) {
        (pt.label == 1 ? minority : majority).push_back(pt);
    }
    
    auto synthetic = balancedSmote_GPU(minority, majority, 5);
    
    // Output first few synthetic samples
    cout << "First 5 synthetic samples:\n";
    for (int i = 0; i < min(5, (int)synthetic.size()); i++) {
        cout << "Sample " << i << ": [";
        for (auto f : synthetic[i].features) cout << f << " ";
        cout << "], Label: " << synthetic[i].label << "\n";
    }
}
/* 
----------------------------------------UNIT TESTING-------------------------------------------------
*/
int main() {
    try {
        /*
        
        // Run tests
        testDistanceKernel();
        testSampleGeneration();
        goldenTest();
        
        // Performance test
        benchmark();
        
        // Visual inspection
        visualizeResults();
        
        */

        // Load data
        string inputFile = "test_datasets/dataset_5_samples_5000_features_30_imbalance_0.95.csv";
        vector<DataPoint> data = readCSV(inputFile);

        // Separate classes
        vector<DataPoint> minorityData, majorityData;
        for (const auto& point : data) {
            if (point.label == 1) minorityData.push_back(point);
            else majorityData.push_back(point);
        }

        // Calculate class distribution
        cout << "Original distribution:\n";
        cout << "  Majority class (0): " << majorityData.size() << " samples\n";
        cout << "  Minority class (1): " << minorityData.size() << " samples\n";

        // Run balanced SMOTE
        int k = 5; // Number of neighbors
        auto syntheticData =  balancedSmote_GPU(minorityData, majorityData, k);

        // Combine data
        vector<DataPoint> balancedData = majorityData;
        balancedData.insert(balancedData.end(), minorityData.begin(), minorityData.end());
        balancedData.insert(balancedData.end(), syntheticData.begin(), syntheticData.end());

        // Verify balance
        int newMinorityCount = minorityData.size() + syntheticData.size();
        cout << "\nBalanced distribution:\n";
        cout << "  Majority class (0): " << majorityData.size() << " samples\n";
        cout << "  Minority class (1): " << newMinorityCount << " samples\n";

        // Save results
        string outputFile = "output_balanced.csv";
        writeCSV(outputFile, balancedData);
        cout << "\nBalanced data saved to: " << outputFile << endl; 
        

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}