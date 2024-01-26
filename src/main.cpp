#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <Eigen/Dense>
#include <numeric>

using namespace Eigen;
using namespace std;
typedef unsigned long long file_position_type;
typedef float ts_type;

struct Node
{
    int start;
    int end;
    VectorXd mean;
    VectorXd farthest_instance;
};

struct Vgs_index
{
    int start;
    int end;
    Node node;
    Vgs_index *left;
    Vgs_index *right;
    Vgs_index(Node node, int start) : node(node), start{start}, end{start} {}
    Vgs_index(Node node, int start, int end) : node(node), start{start}, end{end} {}
};

void read_file(MatrixXd &matrix, char *filename, int vectorSize, int numVectors)
{
    auto *ts = static_cast<ts_type *>(malloc(sizeof(ts_type) * vectorSize));

    FILE *ifile;
    ifile = fopen(filename, "rb");
    if (ifile == nullptr)
    {
        fprintf(stderr, "Error in index.c: File %s not found!\n", filename);
        exit(-1);
    }

    fseek(ifile, 0L, SEEK_END);
    auto sz = (file_position_type)ftell(ifile);
    file_position_type total_records = sz / vectorSize * sizeof(ts_type);
    fseek(ifile, 0L, SEEK_SET);

    if (total_records < numVectors)
    {
        fprintf(stderr, "File %s has only %llu records!\n", filename, total_records);
        exit(-1);
    }

    file_position_type ts_loaded = 0;
    while (ts_loaded < numVectors)
    {

        fread(ts, sizeof(ts_type), vectorSize, ifile);

        for (int i = 0; i < vectorSize; ++i)
        {
            matrix(ts_loaded, i) = ts[i];
        }
        ts_loaded++;
    }
    cout << endl;
}

double circle_distance(Node node, ts_type *query)
{
    int dim = node.mean.cols();
    Eigen::VectorXd diff(dim);
    for (int i = 0; i < dim; i++)
    {
        diff(i) = node.mean(i) - query[i];
    }
    return diff.norm();
}

Vgs_index *query_index(Vgs_index *index, ts_type *query)
{
    if (index == nullptr)
        return nullptr;
    if (index->start != index->end)
    {
        if (circle_distance(index->left->node, query) < circle_distance(index->right->node, query))
            return query_index(index->left, query);
        else
            return query_index(index->right, query);
    }
    else
        return index;
}

void query_file(Vgs_index *index, char *query_filename, int timeseries_size, int num_queries)
{

    FILE *query_file = fopen(query_filename, "rb");
    if (query_file == NULL)
    {
        fprintf(stderr, "Queries file %s not found!\n", query_filename);
        exit(-1);
    }

    fseek(query_file, 0L, SEEK_END);
    file_position_type sz = (file_position_type)ftell(query_file);
    fseek(query_file, 0L, SEEK_SET);
    file_position_type total_records = sz / timeseries_size * sizeof(ts_type);

    fseek(query_file, 0L, SEEK_SET);
    unsigned int offset = 0;

    if (total_records < num_queries)
    {
        fprintf(stderr, "File %s has only %llu records!\n", query_filename, total_records);
        exit(-1);
    }

    unsigned int q_loaded = 0;
    unsigned int ts_length = timeseries_size;

    ts_type *query_ts = static_cast<ts_type *>(malloc(sizeof(ts_type) * ts_length));

    while (q_loaded < num_queries)
    {
        q_loaded++;
        fread(query_ts, sizeof(ts_type), ts_length, query_file);

    }

    free(query_ts);
    fclose(query_file);
}

Vgs_index *merge_nodes(vector<Node> const &vgs, int start, int end)
{
    int start1 = min(vgs[start].start, vgs[end].start);
    int end1 = max(vgs[start].end, vgs[end].end);
    int size1 = (vgs[start].end - vgs[start].start + 1);
    int size2 = (vgs[end].end - vgs[end].start + 1);
    VectorXd mean = ((size1 * vgs[start].mean) + (size2 * vgs[end].mean)) / (size2 + size1);
    VectorXd farthest_element = (vgs[start].mean + vgs[start].farthest_instance).cwiseMax(vgs[end].mean + vgs[end].farthest_instance);
    Node node = {start1, end1, mean, farthest_element};
    return new Vgs_index(node, start, end);
}

Vgs_index *build_index(vector<Node> const &vgs, int start, int end)
{
    if (start == end)
    {
        // Create a leaf node
        return new Vgs_index(vgs[start], start);
    }

    // Create a non-leaf node with the start and end indices
    Vgs_index *root = merge_nodes(vgs, start, end);

    int mid = (start + end) / 2;

    // Recursively build the left and right subtrees
    root->left = build_index(vgs, start, mid);
    root->right = build_index(vgs, mid + 1, end);

    return root;
}


VectorXi column_variance(const MatrixXd &mat)
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

bool compare_vecs(const Eigen::VectorXd &a, const Eigen::VectorXd &b, const Eigen::VectorXi &col_idx, double gap, double min_val)
{
    for (int i = 0; i < a.size(); i++)
    {
        if ((floor(a(col_idx(i)) - min_val) / gap) != (floor(b(col_idx(i)) - min_val) / gap))
        {
            return ((floor(a(col_idx(i)) - min_val) / gap) < (floor(b(col_idx(i)) - min_val) / gap));
        }
    }
    return true;
}

VectorXi sort_mat_rows(MatrixXd &mat, const VectorXi &col_idx, int bins)
{
    double min_val = mat.minCoeff();
    double max_val = mat.maxCoeff();
    double gap = (min_val + max_val) / bins; // 4 represent the number of bins to create for discretizing the matrix values.
    std::vector<int> indices(mat.rows());
    iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](int a, int b)
              { return compare_vecs(mat.row(a), mat.row(b), col_idx, gap, min_val); });

    Eigen::VectorXi ind = Map<Eigen::VectorXi, Eigen::Unaligned>(indices.data(), indices.size());
    return ind;
}

vector<int> row_distance(const MatrixXd &mat, VectorXi col_index, VectorXi row_index, vector<double> &distances)
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
    // static char *filename = "nodataset";
    static char *filename = argv[1]; // data filen name
    int vectorSize = stoi(argv[2]);  // the vector dimension
    int numVectors = stoi(argv[3]);  // number of vectors to load
    int K = stoi(argv[4]);           // number of clusters

    MatrixXd matrix(numVectors, vectorSize);
    // matrix << 2.3, 5.7, 8.1, 3.2, 1.5,
    //     9.4, 4.6, 6.8, 0.9, 7.2,
    //     1.8, 3.5, 7.9, 2.1, 9.7,
    //     4.3, 0.6, 8.7, 5.1, 6.4,
    //     7.6, 2.9, 4.5, 9.2, 1.0;
    read_file(matrix, filename, vectorSize, numVectors);
    // cout << matrix << endl;
    // order the matrix based on the variance
    VectorXi col_index = column_variance(matrix);

    int bins = 4;
    // sort matrows
    VectorXi row_index = sort_mat_rows(matrix, col_index, bins);

    vector<double> distances(numVectors - 1);
    vector<int> idx;
    idx = row_distance(matrix, col_index, row_index, distances);
    vector<Node> vgs;
    leaf_nodes(matrix, idx, col_index, row_index, vgs, K);

    Vgs_index *index = build_index(vgs, 0, vgs.size() - 1);

    // for (const Node &node : vgs)
    // {
    //     std::cout << "Start: " << node.start << "\n";
    //     std::cout << "End: " << node.end << "\n";
    //     std::cout << "Mean: " << node.mean.transpose() << "\n";
    //     std::cout << "Farthest Instance: " << node.farthest_instance.transpose() << "\n";
    //     std::cout << "------------------------\n";
    // }
    // return 0;
}