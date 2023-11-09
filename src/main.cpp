#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <numeric>

using namespace Eigen;
using namespace std;

void readFile(MatrixXd &matrix, std::string const &filename, int vectorSize, int numVectors)
{

    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error: Unable to open file." << endl;
        exit(1);
    }

    vector<VectorXd> vectors;

    string line;
    for (int i = 0; i < numVectors; i++)
    {
        VectorXd vector(vectorSize);
        file.read(reinterpret_cast<char *>(vector.data()), vectorSize * sizeof(float));

        // Check for any read errors
        if (file.fail())
        {
            cerr << "Error: Failed to read data from the file." << endl;
            exit(1);
        }

        vectors.push_back(vector);
    }

    for (int i = 0; i < numVectors; i++)
    {
        matrix.row(i) = vectors[i];
    }
}

VectorXi columnVariance(const MatrixXd &mat)
{
    int numCols = mat.cols();

    VectorXd variances(numCols);
    VectorXd means = mat.colwise().mean();

    for (int i = 0; i < numCols; i++)
    {
        VectorXd centered = mat.col(i) - VectorXd::Constant(mat.rows(), means(i));
        centered = centered.array().square();
        variances(i) = centered.mean();
    }

    std::sort(variances.data(), variances.data() + variances.size(), [](const double &a, const double &b)
              { return a < b; });

    VectorXi idx(variances.size());

    iota(idx.begin(), idx.end(), 0);

    sort(idx.begin(), idx.end(), [&variances](size_t i1, size_t i2)
         { return variances[i1] < variances[i2]; });
    return idx;
}

void permuteMat(MatrixXd &mat, const VectorXi &idx, bool cols = true, bool inverse = false)
{

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm;
    perm.indices() = idx;
    // Permutate cols
    if (cols)
    {
        if (inverse)
        {
            mat = mat * perm.inverse();
            return;
        }
        mat = mat * perm;
        return;
    }
    if (inverse)
    {
        mat = perm.inverse() * mat;
        return;
    }
    mat = perm * mat;
}

bool customComparison(const Eigen::VectorXd &a, const Eigen::VectorXd &b, int gap = 1)
{
    for (int i = 0; i < a.size(); i++)
    {
        if (floor(a(i)/gap) != floor(b(i)/gap))
            return false;
    }
    return true;
}

void sortMatRows(MatrixXd &mat)
{

    std::vector<int> indices(mat.rows());
    iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](int a, int b)
                     { return customComparison(mat.row(a), mat.row(b)); });

    Eigen::VectorXi ind = Map<Eigen::VectorXi, Eigen::Unaligned>(indices.data(), indices.size());

    permuteMat(mat, ind, false);
}

double rowDistance(const MatrixXd &mat, vector<double> &distances)
{
    for (int i = 0; i < mat.rows() - 1; i++)
    {
        Eigen::VectorXd diff = mat.row(i) - mat.row(i + 1);
        distances[i] = diff.norm();
    }
}

int main(int argc, char *argv[])
{

    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <filename> <dimension> <size>" << std::endl;
        return 1;
    }
    string filename = argv[1];      // data filen name
    int vectorSize = stoi(argv[2]); // the vector dimension
    int numVectors = stoi(argv[3]); // number of vectors to load
    int K = stoi(argv[4]);          // number of clusters

    MatrixXd matrix(numVectors, vectorSize);

    readFile(matrix, filename, vectorSize, numVectors);

    // order the matrix based on the variance
    VectorXi indexes = columnVariance(matrix);

    // order the mat based on rows
    sortMatRows(matrix);

    // calcuate the row distances.
    vector<double> distances(matrix.rows() - 1);
    rowDistance(matrix, distances);

    // Calculate the clusters delimeters
    vector<double> clusters(K);
    vector<int> idx(distances.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&distances](size_t i1, size_t i2)
         { return distances[i1] < distances[i2]; });
    std::copy(idx.end() - K, idx.end(), clusters.begin());



    
    vector<pair<int, int>> Nodes(K);
    Nodes.push_back({0, idx[0]});

    // Adding the rest of the Nodes
    for (int i = 0; i < K - 1; i++) {
        Nodes.push_back({idx[i], idx[i + 1]});
    }

    // sample data testing
    Eigen::MatrixXd mat(3, 3); 
    mat << 1, 2, 3,
        4, 5, 6,
        2, 8, 9;

    VectorXi indexes = columnVariance(mat);

    permuteMat(mat, indexes);
    cout << mat << endl;

    sortMatRows(mat);
    cout << " \n sorted rows \n"
         << mat << endl;

    return 0;
}
