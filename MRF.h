#ifndef MRF_H
#define MRF_H

#include <vector>
#include <string>
using namespace std;

template<class T>
struct index_cmp {
  index_cmp(const T arr) : arr(arr) {}
  bool operator()(const int a, const int b) const
  { return arr[a] < arr[b]; }
  const T arr;
};

struct Node {
  vector<vector<float>* > inputs;
  vector<vector<float>* > outputs;
  vector<float>* mean_output;
  int split_variable;
  float split_value;
  Node* child1;
  Node* child2;
  float error;
  int tree_index;

  Node() {
    child1 = NULL;
    child2 = NULL;
    error = -1;
  }
  
  ~Node() {
    delete child1;
    delete child2;
  }
};

extern "C++" {
struct Split {
  int variable_index;
  float split_value;
};
}

class MRF {
  
  private:
    int num_inputs;
    int num_input_vars;
    int num_output_vars;
    int num_ensembles;
    int mtry;
    int min_terminal_size;
    bool update;
    int iterations;
    int number_to_destroy;
    vector<vector<float>* >* all_inputs;
    vector<float> input_min;
    vector<float> input_max;
    vector<bool>* discrete;
    vector<vector<float>* > all_outputs;
    vector<vector<float>* >* all_outputs_unnorm;
    vector<float> input_means;
    vector<float> input_devs;
    vector<float> output_means;
    vector<float> output_devs;
    vector<Node*> roots;
    vector<vector<int>* > OOB_inputs_for_trees;
    vector<vector<Node*>* > OOB_trees_for_inputs;
    vector<vector<float>* > predictions;
    vector<float> prediction_errors;

    void create_tree(Node* root);
    void make_leaf(Node* root);
    float get_node_impurity(Node* node);
    float get_node_impurity_sum(vector<float>& sum, Node* node, int start, int end,
                                vector<int>& input_indices);
    void split_node(Node* node, int var_index, float var_value, Node* child1,
                    Node* child2);
    void generate_tree();
    static void tokenize(string line, vector<string>& tokens);
    void print_node(ofstream& file, Node* node, bool left, int level,
                   int split_index, float split_value);
    void normalize_output();
    void determine_predictions();
    void determine_MSEs();
    float MSE(vector<float>* actual, vector<float>* predicted);
    bool perform_best_split(Node* root);
    inline float ranf();
    float random_normal(float mean, float dev);
    void determine_variable_stats(vector<vector<float>* >* matrix,
                                  vector<float>& means,
                                  vector<float>& deviations);
    void determine_predictions_errors();

  public:
    MRF(vector<vector<float>* >* all_inputs,
        vector<bool>* discrete,
        vector<vector<float>* >* all_outputs,
        char* output_dir,
        bool log,
        int num_ensembles,
        int mtry,
        int min_terminal_size);
    ~MRF();
    void output_estimate(vector<float>* input, vector<float>* output,
                         vector<Node*>& trees);
    static void read_data(const char* filename, vector<vector<float>* >& matrix,
                         vector<bool>* discrete);
    void write_output(const char* filename,
                            vector<vector<float>* >& matrix);
    void write_predictions_errors(const char* filename_predictions,
                                  const char* filename_errors,
                                  const char* norm_filename_predictions=NULL,
                                  const char* norm_filename_errors=NULL);
    void write_MSEs(vector<vector<float>* >& predictions_unnorm,
                    const char* filename, const char* norm_filename=NULL);
    void print_trees(const char* filename);
    void print_OOB_trees_for_inputs(const char* filename);
    void print_OOB_inputs_for_trees(const char* filename);
    void print_int_vector(vector<int>& vec);
    void print_float_vector(vector<float>& vec);
    void write_feature_distribution(const char* filename);
    void calculate_var_importance();
    float calculate_mean(vector<float>& values);
    float calculate_standard_deviation(vector<float>& values);
    void calculate_tree_errors(vector<float>& tree_errors);
    // static PyObject* MRF_test(PyObject* self, PyObject* args);
};

#endif /* MRF_H */
