#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <numeric>

using namespace Eigen;
using namespace std;

struct Nodes{
        int start;
        int end;
        VectorXd mean;
        VectorXd farthest_instance;
    };

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

VectorXi columnVariance(const MatrixXd &mat) {
    int numCols = mat.cols();

    VectorXd variances(numCols);
    VectorXd means = mat.colwise().mean();

    for (int i = 0; i < numCols; i++) {
        ArrayXd centered = mat.col(i).array() - means(i);
        centered = centered.square();
        variances(i) = centered.mean();
    }

    // Sort the indices based on variances
    VectorXi idx(variances.size());
    iota(idx.data(), idx.data() + idx.size(), 0);
    sort(idx.data(), idx.data() + idx.size(), [&variances](int i1, int i2) {
        return variances(i1) < variances(i2);
    });

    return idx;
}
/*
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
*/
bool customComparison(const Eigen::VectorXd &a, const Eigen::VectorXd &b,const Eigen::VectorXi &col_idx, int gap = 1)
{
    for (int i = 0; i < a.size(); i++)
    {
        if (floor(a(col_idx(i))/gap) != floor(b(col_idx(i))/gap)){
            return (floor(a(col_idx(i))/gap) < floor(b(col_idx(i))/gap));
            }
    }
    return true;
}

VectorXi sortMatRows(MatrixXd &mat, const VectorXi &col_idx)
{

    std::vector<int> indices(mat.rows());
    iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](int a, int b)
              { return customComparison(mat.row(a), mat.row(b), col_idx); });

    Eigen::VectorXi ind = Map<Eigen::VectorXi, Eigen::Unaligned>(indices.data(), indices.size());
    return ind;
}

void rowDistance(const MatrixXd &mat, vector<double> &distances)
{
    for (int i = 0; i < mat.rows() - 1; i++)
    {
        Eigen::VectorXd diff = mat.row(i) - mat.row(i + 1);
        distances[i] = diff.norm();
    }
}

void segments(const vector<double> &distances, int k, int numVectors, vector<int> &clusters){
    vector<int> idx(distances.size());
    iota(idx.begin(), idx.end(), 0);
    sort(idx.begin(), idx.end(), [&distances](size_t i1, size_t i2)
         { return distances[i1] < distances[i2]; });

    clusters.push_back(-1);
    for (int i = 0; i < k; i++)
    {
        clusters.push_back(idx[idx.size()-(k-i)]);
    }
    clusters.push_back(numVectors-1);
}

void leaf_nodes(const MatrixXd &mat, vector<int> &ind, vector<Nodes> &vgs)
{

    for (int i = 0; i < ind.size() - 1; i++)
    {

        VectorXd meanVector = mat.block(ind[i], 0, ind[i + 1] - ind[i] + 1, mat.cols()).colwise().mean();

        // Calculate the Euclidean norms of each row
        VectorXd norms = (mat.block(ind[i], 0, ind[i + 1] - ind[i] + 1, mat.cols()).rowwise() - meanVector.transpose()).rowwise().norm();

        // Find the index of the row with the maximum Euclidean norm
        int ans = norms[0];
        int arg = 0;
        int normSize = norms.size();
        for (int j = 0; j < normSize; j++)
        {
            if(norms[j] > ans){
                ans = norms[j];
                arg = j;
            }
        }
        vgs.push_back({ind[i] + 1, ind[i+1] + 1, meanVector, mat.row(ind[i]+arg)});
    }
}

int main(int argc, char *argv[])
{

    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <filename> <dimension> <size> <K>" << std::endl;
        return 1;
    }
    string filename = argv[1];      // data filen name
    int vectorSize = stoi(argv[2]); // the vector dimension
    int numVectors = stoi(argv[3]); // number of vectors to load
    int K = stoi(argv[4]);          // number of clusters

    MatrixXd matrix(numVectors, vectorSize);
    matrix << 2.3, 5.7, 8.1, 3.2, 1.5,
            9.4, 4.6, 6.8, 0.9, 7.2,
            1.8, 3.5, 7.9, 2.1, 9.7,
            4.3, 0.6, 8.7, 5.1, 6.4,
            7.6, 2.9, 4.5, 9.2, 1.0;

    // readFile(matrix, filename, vectorSize, numVectors);

    // order the matrix based on the variance
    VectorXi col_index = columnVariance(matrix);

    // sort matrows 
    VectorXi row_index = sortMatRows(matrix, col_index);

    // test
    // std::vector<int> indices(matrix.rows());
    // iota(indices.begin(), indices.end(), 0);
    // Eigen::VectorXi ind = Map<Eigen::VectorXi, Eigen::Unaligned>(indices.data(), indices.size());
    // VectorXi row_index = sortMatRows(matrix, ind);
    cout << row_index ;

    return 0;
}
