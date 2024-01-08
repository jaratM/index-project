#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <numeric>

using namespace Eigen;
using namespace std;

struct Node
{
    int start;
    int end;
    VectorXd mean;
    VectorXd farthest_instance;
};

struct Vgs_index
{
    Node node;
    Vgs_index *left;
    Vgs_index *right;
    Vgs_index(Node node) : node(node), left(nullptr), right(nullptr) {}
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

Node merge_nodes(Node n1, Node n2)
{
    int start = min(n1.start, n2.start);
    int end = max(n1.end, n2.end);
    VectorXd mean = (((n1.end - n1.start + 1) * n1.mean) + ((n2.end - n2.start + 1) * n2.mean)) / ((n2.end - n2.start + 1) + (n1.end - n1.start + 1));
    VectorXd farthest_element = (n1.mean + n1.farthest_instance).cwiseMax(n2.mean + n2.farthest_instance);
    return {start, end, mean, farthest_element};
}

Vgs_index *build_index(vector<Node> const &vgs, int start, int end)
{
    if (start == end)
    {
        // Create a leaf node
        return new Vgs_index(vgs[start]);
    }

    // Create a non-leaf node with the start and end indices
    Node merged_nodes = merge_nodes(vgs[start], vgs[end]);

    Vgs_index *root = new Vgs_index(merged_nodes);

    int mid = (start + end) / 2;

    // Recursively build the left and right subtrees
    root->left = build_index(vgs, start, mid);
    root->right = build_index(vgs, mid + 1, end);

    return root;
}

VectorXi columnVariance(const MatrixXd &mat)
{
    int numCols = mat.cols();

    VectorXd variances(numCols);
    VectorXd means = mat.colwise().mean();

    for (int i = 0; i < numCols; i++)
    {
        ArrayXd centered = mat.col(i).array() - means(i);
        centered = centered.square();
        variances(i) = centered.mean();
    }

    // Sort the indices based on variances
    VectorXi idx(variances.size());
    iota(idx.data(), idx.data() + idx.size(), 0);
    sort(idx.data(), idx.data() + idx.size(), [&variances](int i1, int i2)
         { return variances(i1) > variances(i2); });
    return idx;
}

// void permuteMat(MatrixXd &mat, const VectorXi &idx, bool cols = true, bool inverse = false)
// {
//     Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm;
//     perm.indices() = idx;
//     // Permutate cols
//     if (cols)
//     {
//         if (inverse)
//         {
//             mat = mat * perm.inverse();
//             return;
//         }
//         mat = mat * perm;
//         return;
//     }
//     if (inverse)
//     {
//         mat = perm * mat;
//         return;
//     }
//     mat = perm.inverse() * mat;
// }

bool customComparison(const Eigen::VectorXd &a, const Eigen::VectorXd &b, const Eigen::VectorXi &col_idx, double gap)
{
    for (int i = 0; i < a.size(); i++)
    {
        if (floor(a(col_idx(i)) / gap) != floor(b(col_idx(i)) / gap))
        {
            return (floor(a(col_idx(i)) / gap) < floor(b(col_idx(i)) / gap));
        }
    }
    return true;
}

VectorXi sortMatRows(MatrixXd &mat, const VectorXi &col_idx)
{
    double min_val = mat.minCoeff();
    double max_val = mat.maxCoeff();
    double bin = (min_val  + max_val)/4;  // 4 represent the number of bins to create for discretizing the matrix values.
    std::vector<int> indices(mat.rows());
    iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](int a, int b)
              { return customComparison(mat.row(a), mat.row(b), col_idx, bin); });

    Eigen::VectorXi ind = Map<Eigen::VectorXi, Eigen::Unaligned>(indices.data(), indices.size());
    return ind;
}

vector<int> rowDistance(const MatrixXd &mat, VectorXi col_index, VectorXi row_index, vector<double> &distances)
{
    for (int i = 0; i < mat.rows() - 1; i++)
    {
        Eigen::VectorXd diff(mat.cols());

        for (int j = 0; j < mat.cols(); j++)
        {
            diff(j) = mat(row_index[i], col_index[j]) - mat(row_index[i + 1], col_index[j]);
        }
        distances[i] = diff.norm();
    }

    vector<int> idx(distances.size());
    iota(idx.data(), idx.data() + idx.size(), 0);
    sort(idx.data(), idx.data() + idx.size(), [&distances](int i1, int i2)
         { return distances[i1] > distances[i2]; });
    return idx;
}

void leaf_nodes(const MatrixXd &mat, vector<int> &idx, VectorXi col_index, VectorXi row_index, vector<Node> &vgs, int K)
{
    std::vector<int> ind(K);
    std::copy(idx.begin(), idx.begin() + K, ind.begin());
    std::sort(ind.begin(), ind.end());

    ind.insert(ind.begin(), -1);
    ind.push_back(mat.rows() - 1);

    for (int i = 1; i < ind.size(); i++)
    {
        VectorXd meanVector(mat.cols());
        for (int col = 0; col < mat.cols(); col++)
        {
            meanVector(col) = 0;
            for (int j = ind[i - 1] + 1; j < ind[i] + 1; j++)
            {
                meanVector(col) = meanVector(col) + mat(row_index(j), col_index(col));
            }
            meanVector(col) = double(meanVector(col) / (ind[i] - ind[i - 1]));
        }

        int subrows_len = (ind[i] - ind[i - 1]);
        VectorXd norms(subrows_len);

        for (int j = 0; j < subrows_len; j++)
        {
            VectorXd tmp(mat.cols());
            for (int k = 0; k < mat.cols(); k++)
            {
                tmp(k) = mat(row_index(j + ind[i - 1] + 1), col_index(k)) - meanVector(k);
            }
            norms(j) = tmp.norm();
        }

        int ans = norms[0];
        int arg = 0;
        int normSize = norms.size();
        for (int j = 0; j < normSize; j++)
        {
            if (norms[j] > ans)
            {
                ans = norms[j];
                arg = j;
            }
        }
        VectorXd farthest_element(mat.cols());
        for (int j = 0; j < mat.cols(); j++)
        {
            farthest_element(j) = mat(row_index(ind[i - 1] + 1 + arg), col_index(j));
        }
        vgs.push_back({ind[i - 1] + 1, ind[i] + 1, meanVector, farthest_element});
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

    vector<double> distances(numVectors - 1);
    vector<int> idx;
    idx = rowDistance(matrix, col_index, row_index, distances);
    vector<Node> vgs;

    leaf_nodes(matrix, idx, col_index, row_index, vgs, K);

    Vgs_index *index = build_index(vgs, 0, vgs.size() - 1);

    for (const Node &node : vgs)
    {
        std::cout << "Start: " << node.start << "\n";
        std::cout << "End: " << node.end << "\n";
        std::cout << "Mean: " << node.mean.transpose() << "\n";
        std::cout << "Farthest Instance: " << node.farthest_instance.transpose() << "\n";
        std::cout << "------------------------\n";
    }

    return 0;
}