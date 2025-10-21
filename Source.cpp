#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <sstream>
#include <memory>
#include <iomanip>

struct Image {
    int width, height;
    std::vector<uint8_t> pixels;

    Image(int w = 0, int h = 0) : width(w), height(h), pixels(w* h, 0) {}

    uint8_t& at(int row, int col) { return pixels[row * width + col]; }
    const uint8_t& at(int row, int col) const { return pixels[row * width + col]; }
};

struct FilterResult {
    std::string imageName;
    double noiseLevel;
    int filterSize;
    double mse;
    double psnr;
    double ssim;

    FilterResult() : imageName(""), noiseLevel(0.0), filterSize(0),
        mse(0.0), psnr(0.0), ssim(0.0) {
    }
};

class PGMLoader {
public:
    static Image load(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) throw std::runtime_error("Cannot open: " + filename);

        std::string magic;
        file >> magic;
        if (magic != "P2") throw std::runtime_error("Not P2 format: " + filename);

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') continue;

            std::stringstream ss(line);
            int width, height, maxval;
            if (ss >> width >> height) {
                file >> maxval;

                Image img(width, height);
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        int pixel;
                        if (!(file >> pixel)) {
                            throw std::runtime_error("Error reading pixel data");
                        }
                        img.at(i, j) = static_cast<uint8_t>(std::max(0, std::min(255, pixel)));
                    }
                }

                return img;
            }
        }

        throw std::runtime_error("Invalid PGM file format");
    }

    static void save(const std::string& filename, const Image& img) {
        std::ofstream file(filename);
        if (!file) throw std::runtime_error("Cannot create: " + filename);

        file << "P2\n" << img.width << " " << img.height << "\n255\n";

        for (int i = 0; i < img.height; i++) {
            for (int j = 0; j < img.width; j++) {
                file << static_cast<int>(img.at(i, j));
                if (j < img.width - 1) file << " ";
            }
            file << "\n";
        }
    }
};

class NoiseAdder {
    std::mt19937 gen;

public:
    NoiseAdder() {
        std::random_device rd;
        gen.seed(rd());
    }

    Image addSaltPepper(const Image& img, double noise_level = 0.05) {
        Image noisy = img;
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        for (int i = 0; i < img.height; i++) {
            for (int j = 0; j < img.width; j++) {
                double r = dist(gen);
                if (r < noise_level / 2) noisy.at(i, j) = 255;
                else if (r < noise_level) noisy.at(i, j) = 0;
            }
        }
        return noisy;
    }
};

class MeanFilter {
public:
    Image apply(const Image& img, int window_size) {
        if (window_size % 2 == 0) throw std::invalid_argument("Window size must be odd");

        Image result(img.width, img.height);
        int offset = window_size / 2;

        for (int i = 0; i < img.height; i++) {
            for (int j = 0; j < img.width; j++) {
                int sum = 0, count = 0;

                for (int di = -offset; di <= offset; di++) {
                    for (int dj = -offset; dj <= offset; dj++) {
                        int ni = i + di, nj = j + dj;
                        if (ni >= 0 && ni < img.height && nj >= 0 && nj < img.width) {
                            sum += img.at(ni, nj);
                            count++;
                        }
                    }
                }
                result.at(i, j) = static_cast<uint8_t>(sum / std::max(1, count));
            }
        }
        return result;
    }
};

class ImageMetrics {
public:
    static double calculateMSE(const Image& orig, const Image& filtered) {
        if (orig.width != filtered.width || orig.height != filtered.height) {
            throw std::invalid_argument("Images must have same dimensions for MSE calculation");
        }

        double mse = 0.0;
        int total_pixels = orig.width * orig.height;

        for (int i = 0; i < orig.height; i++) {
            for (int j = 0; j < orig.width; j++) {
                double diff = static_cast<double>(orig.at(i, j)) - static_cast<double>(filtered.at(i, j));
                mse += diff * diff;
            }
        }
        return mse / total_pixels;
    }

    static double calculatePSNR(const Image& orig, const Image& filtered) {
        double mse = calculateMSE(orig, filtered);

        if (mse <= 1e-10) return 100.0;

        double psnr = 10.0 * log10(255.0 * 255.0 / mse);

        if (std::isnan(psnr) || std::isinf(psnr)) {
            return 0.0;
        }

        return psnr;
    }

    static double calculateSSIM(const Image& img1, const Image& img2) {
        if (img1.width != img2.width || img1.height != img2.height) {
            throw std::invalid_argument("Images must have same dimensions for SSIM calculation");
        }

        const double C1 = 6.5025, C2 = 58.5225;
        const int total_pixels = img1.width * img1.height;

        double mu1 = 0.0, mu2 = 0.0;
        for (int i = 0; i < img1.height; i++) {
            for (int j = 0; j < img1.width; j++) {
                mu1 += img1.at(i, j);
                mu2 += img2.at(i, j);
            }
        }
        mu1 /= total_pixels;
        mu2 /= total_pixels;

        double sigma1_sq = 0.0, sigma2_sq = 0.0, sigma12 = 0.0;
        for (int i = 0; i < img1.height; i++) {
            for (int j = 0; j < img1.width; j++) {
                double diff1 = img1.at(i, j) - mu1;
                double diff2 = img2.at(i, j) - mu2;
                sigma1_sq += diff1 * diff1;
                sigma2_sq += diff2 * diff2;
                sigma12 += diff1 * diff2;
            }
        }

        sigma1_sq /= (total_pixels - 1);
        sigma2_sq /= (total_pixels - 1);
        sigma12 /= (total_pixels - 1);

        double numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2);
        double denominator = (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2);

        if (denominator == 0.0) {
            return 1.0;
        }

        double ssim = numerator / denominator;

        return std::max(0.0, std::min(1.0, ssim));
    }
};

class CSVWriter {
public:
    static void saveResultsToCSV(const std::vector<FilterResult>& results, const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening CSV file: " << filename << std::endl;
            return;
        }

        file << "\xEF\xBB\xBF";

        file << "ImageName;NoiseLevel;FilterSize;MSE;PSNR;SSIM" << std::endl;

        for (const auto& result : results) {
            file << "\"" << result.imageName << "\";"
                << std::fixed << std::setprecision(4) << result.noiseLevel << ";"
                << result.filterSize << ";"
                << std::fixed << std::setprecision(2) << result.mse << ";"
                << std::fixed << std::setprecision(2) << result.psnr << ";"
                << std::fixed << std::setprecision(4) << result.ssim << std::endl;
        }

        file.close();
        std::cout << "Results saved to: " << filename << std::endl;
    }
};

bool createDirectory(const std::string& dirname) {
#ifdef _WIN32
    return system(("mkdir " + dirname).c_str()) == 0;
#else
    return system(("mkdir -p " + dirname).c_str()) == 0;
#endif
}

bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

std::vector<std::string> findPGMFiles() {
    std::vector<std::string> pgmFiles;
    
    for (const auto& name : {"capibara.pgm", "dog.pgm", "Hedgehog.pgm", "kotik.pgm", "minipig.pgm"}) {
        std::string path = "Image/" + std::string(name);
        if (fileExists(path)) {
            pgmFiles.push_back(path);
        }
    }
    
    if (pgmFiles.empty()) throw std::runtime_error("No PGM images found");
    return pgmFiles;
}

class ImageProcessor {
    NoiseAdder noise_adder;
    MeanFilter filter;
    ImageMetrics comparator;
    std::vector<FilterResult> allResults;

public:
    void processAllImages() {
        if (!createDirectory("results")) {
            std::cerr << "Could not create results directory" << std::endl;
        }

        std::vector<std::string> image_files = findPGMFiles();
        std::vector<int> filter_sizes = { 3, 5, 7 };
        const double noise_level = 0.05;

        std::cout << "Processing " << image_files.size() << " images..." << std::endl;

        allResults.clear();

        for (const auto& file_path : image_files) {
            processSingleImage(file_path, filter_sizes, noise_level);
        }

        CSVWriter::saveResultsToCSV(allResults, "results/denoising_results.csv");

        std::cout << "Processing complete. Results saved to: results/denoising_results.csv" << std::endl;
    }

private:
    void processSingleImage(const std::string& file_path,
        const std::vector<int>& filter_sizes,
        double noise_level) {

        try {
            Image original = PGMLoader::load(file_path);

            std::string name = file_path;
            size_t lastSlash = file_path.find_last_of("/\\");
            if (lastSlash != std::string::npos) {
                name = file_path.substr(lastSlash + 1);
            }
            size_t dotPos = name.find_last_of(".");
            if (dotPos != std::string::npos) {
                name = name.substr(0, dotPos);
            }

            Image noisy = noise_adder.addSaltPepper(original, noise_level);
            PGMLoader::save("results/" + name + "_noisy.pgm", noisy);

            for (int size : filter_sizes) {
                try {
                    Image filtered = filter.apply(noisy, size);

                    double mse_value = comparator.calculateMSE(original, filtered);
                    double psnr_value = comparator.calculatePSNR(original, filtered);
                    double ssim_value = comparator.calculateSSIM(original, filtered);

                    FilterResult result;
                    result.imageName = name;
                    result.noiseLevel = noise_level;
                    result.filterSize = size;
                    result.mse = mse_value;
                    result.psnr = psnr_value;
                    result.ssim = ssim_value;

                    allResults.push_back(result);

                    PGMLoader::save("results/" + name + "_filter" +
                        std::to_string(size) + ".pgm", filtered);

                }
                catch (const std::exception& e) {
                    std::cerr << "Error with filter size " << size << " for image " << name << ": " << e.what() << std::endl;
                }
            }

            PGMLoader::save("results/" + name + "_original.pgm", original);
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing image " << file_path << ": " << e.what() << std::endl;
            throw;
        }
    }
};

int main() {
    std::cout << " Image Denoiser " << std::endl;

    try {
        ImageProcessor processor;
        processor.processAllImages();
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}