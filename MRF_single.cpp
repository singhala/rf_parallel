/* Usage:
   ./MRF --output_dir directoryname/ --input_file filename --actual_file \
   filename [--num_trees ##] [--log_after_every_tree]
*/

// #include <Python.h>
#include <vector>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <functional>
#include <numeric>
#include <list>
#include <string>
#include <cstring>
#include <time.h>
#include "MRF_single.h"

#define DIVISIONS 5

MRF::MRF(vector<vector<float>* >* inputs,
         vector<bool>* discrete,
         vector<vector<float>* >* outputs,
         char* output_dir,
         char* read_file,
         bool log=false,
         bool permute=false,
         int num_ensembles=1000,
         int mtry=0,
         int min_terminal_size=5) {
  this->num_ensembles = num_ensembles;
  this->min_terminal_size = min_terminal_size;
  this->all_inputs = inputs;
  this->discrete = discrete;
  this->all_outputs_unnorm = outputs;
  this->num_inputs = all_inputs->size();
  this->num_input_vars = all_inputs->at(0)->size();
  this->num_output_vars = all_outputs_unnorm->at(0)->size();
  this->output_dir = output_dir;
  this->log_after = log;
  
  if (permute) {
    cout << "Permuting" << endl;
    random_shuffle(outputs->begin(), outputs->end());
  }

  if (mtry == 0) { // if mtry not specified
    this->mtry = static_cast<int>(sqrt(num_input_vars));
  } else {
    this->mtry = mtry;
  }

  if (num_inputs != all_outputs_unnorm->size()) {
    cout << "Different numbers of inputs and outputs" << endl;
  }
  // determine min and max of each variable
  vector<vector<float>* >::iterator it;
  fprintf(stdout, "Num input vars: %d\n", num_input_vars);
  for (int i = 0; i < num_input_vars; i++) {
    float min_value = 0;
    float max_value = 0;
    for (it = all_inputs->begin(); it != all_inputs->end(); it++) {
      min_value = min(min_value, (*it)->at(i));
      max_value = max(max_value, (*it)->at(i));
    }
    input_min.push_back(min_value);
    input_max.push_back(max_value);
    // cout << i << ": " << min_value << " " << max_value << endl;
  }

  determine_variable_stats(all_outputs_unnorm, output_means, output_devs);
  determine_variable_stats(all_inputs, input_means, input_devs);
  normalize_output();
  
  for (int i = 0; i < num_inputs; i++) {
    vector<Node*>* OOB_trees = new vector<Node*>;
    OOB_trees_for_inputs.push_back(OOB_trees);
  }
  
  fprintf(stdout,
          "Generating forest with %d inputs, %d input vars, %d output vars\n",
          num_inputs, num_input_vars, num_output_vars);
  if (read_file == NULL) {
    for (int i = 0; i < num_ensembles; i++) {
      generate_tree();
      if (log) {
        log_tree(i);
      }
    }
  } else {
    read_trees(read_file);
  }
  cout << "Generated forest" << endl;
  if (log) {
    determine_predictions_errors();
    cout << "Determined predictions and errors" << endl;
    calculate_var_importance();
    cout << "Determined variable importance" << endl;
  }
}

void MRF::log_tree(int i) {
  stringstream out;
  out << i;
  string iteration = out.str();
  determine_predictions_errors();
  cout << "Determined predictions and errors" << endl;
  string out_str(output_dir);
  string predict_filename = out_str+"predict_"+iteration+".txt";
  string mse_filename = out_str+"MSE_" +iteration+".txt";
  cout << predict_filename << endl;
  cout << mse_filename << endl;
  write_predictions_errors(predict_filename.c_str(),
                           mse_filename.c_str());
  cout << "Wrote predictions and errors" << endl;
} 

MRF::~MRF() {
  vector<Node*>::iterator it;
  for (it = roots.begin(); it != roots.end(); it++) {
    delete *it;
  }
  vector<vector<float>* >::iterator it2;
  for (it2 = all_inputs->begin(); it2 != all_inputs->end(); it2++) {
    delete *it2;
  }
  for (it2 = all_outputs.begin(); it2 != all_outputs.end(); it2++) {
    delete *it2;
  }
  for (it2 = all_outputs_unnorm->begin(); it2 != all_outputs_unnorm->end();
       it2++) {
    delete *it2;
  }
  for (it2 = predictions.begin(); it2 != predictions.end(); it2++) {
    delete *it2;
  }
  vector<vector<int>* >::iterator it3;
  for (it3 = OOB_inputs_for_trees.begin(); it3 != OOB_inputs_for_trees.end(); it3++) {
    delete *it3;
  }
  vector<vector<Node*>* >::iterator it4;
  for (it4 = OOB_trees_for_inputs.begin(); it4 != OOB_trees_for_inputs.end(); it4++) {
    delete *it4;
  }
}

void MRF::normalize_output() {
  for (int i = 0; i < num_inputs; i++) {
    vector<float>* norm_values = new vector<float>(num_output_vars, 0);
    all_outputs.push_back(norm_values);
  }
  for (int i = 0; i < num_output_vars; i++) {
    float mean = output_means.at(i);
    float deviation = output_devs.at(i);
    // printf("Output %d: Mean %.4f, Std Dev: %.4f\n", i, mean, deviation);
    for (int j = 0; j < num_inputs; j++) {
      (all_outputs[j])->at(i) = ((all_outputs_unnorm->at(j))->at(i) - mean) /
          deviation;
    }
  }
}

void MRF::generate_tree() {
  // create subsets of the input with replacement, each in its own root node
  Node* root = new Node;
  vector<bool> used_here(num_inputs, false);
  for (int j = 0; j < num_inputs; j++) {
    int index = rand() % num_inputs;
    if (!(used_here[index])) {
      used_here[index] = true;
      root->inputs.push_back(all_inputs->at(index));
      root->outputs.push_back(all_outputs.at(index));
    }
  }
   
  // determine the inputs that are OOB for this tree
  vector<int>* OOB_inputs = new vector<int>;
  for (int j = 0; j < num_inputs; j++) {
    if (!(used_here[j])) {
      OOB_inputs->push_back(j);
      OOB_trees_for_inputs.at(j)->push_back(root);
    }
  }
  OOB_inputs_for_trees.push_back(OOB_inputs);
  root->tree_index = roots.size();
  roots.push_back(root);
  printf("Ensemble\n");
  time_t start, end;
  time(&start);
  create_tree(roots.at(roots.size()-1));
  cout << "Created tree " << roots.size() << endl;
  time(&end);
  printf("%.2lf seconds taken to create tree\n", difftime(end, start));
}

void MRF::calculate_var_importance() {
  // calculate MSE on OOB for each tree
  vector<float> tree_errors;
  calculate_tree_errors(tree_errors);
  vector<float> var_importance;
  
  for (int i = 0; i < num_input_vars; i++) {
    // permute one variable and do the same
    cout << "Permuting variable " << i << endl;
    vector<float> var_values;
    for (int j = 0; j < num_inputs; j++) {
      var_values.push_back(all_inputs->at(j)->at(i));
    }
    vector<float> var_values_permuted(var_values);
    random_shuffle(var_values_permuted.begin(), var_values_permuted.end());
    for (int j = 0; j < num_inputs; j++) {
      all_inputs->at(j)->at(i) = var_values_permuted.at(j);
    }
    vector<float> tree_errors_permuted;
    calculate_tree_errors(tree_errors_permuted);
    
    // find average difference, normalized by standard error
    vector<float> MSE_differences;
    for (int j = 0; j < tree_errors.size(); j++) {
      MSE_differences.push_back(tree_errors_permuted[j] - tree_errors[j]);
    }
    float mean = calculate_mean(MSE_differences);
    float deviation = calculate_standard_deviation(MSE_differences);
    float importance = 0 ? deviation == 0 : mean/deviation;
    var_importance.push_back(importance);
    // reset this variable
    for (int j = 0; j < num_inputs; j++) {
      all_inputs->at(j)->at(i) = var_values.at(j);
    }
  }

  vector<vector<float>* > matrix;
  matrix.push_back(&var_importance);
  string out_str(output_dir);
  string var_string = out_str + "var_importance.txt";
  write_output(var_string.c_str(), matrix);
}

float MRF::calculate_mean(vector<float>& values) {
  return accumulate(values.begin(), values.end(), 0.0f) / values.size();
} 

float MRF::calculate_standard_deviation(vector<float>& values) {
  float mean = calculate_mean(values);
  vector<float> zero_mean(values);
  transform(zero_mean.begin(), zero_mean.end(), zero_mean.begin(),
            bind2nd(minus<float>(), mean));
  // find the standard deviation
  float deviation = inner_product(zero_mean.begin(), zero_mean.end(),
                                  zero_mean.begin(), 0.0f );
  return sqrt(deviation / (values.size() - 1 ));
}

void MRF::calculate_tree_errors(vector<float>& tree_errors) {
  for (int i = 0; i < roots.size(); i++) {
    Node* root = roots.at(i);
    vector<int>* OOB_inputs = OOB_inputs_for_trees.at(i);
    vector<Node*> this_tree;
    this_tree.push_back(root);
    float total_MSE = 0;
    for (int j = 0; j < OOB_inputs->size(); j++) {
      int input_index = OOB_inputs->at(j);
      vector<float>* input = all_inputs->at(input_index);
      vector<float> predicted_output(num_output_vars, 0);
      output_estimate(input,  &predicted_output, this_tree);
      total_MSE += MSE(all_outputs.at(input_index), &predicted_output);
    }
    float error = total_MSE / float(OOB_inputs->size());
    tree_errors.push_back(error);
  }
}

void MRF::create_tree(Node* root) {
  if (root->inputs.size() <= min_terminal_size) {
    // cout << "Reached leaf of size " << root->inputs.size() << endl;
    // don't split past min_terminal_size
    make_leaf(root);
    return;
  }
  // cout << "Splitting node" << endl;
  if (perform_best_split(root)) {
    // better split found
    // create the tree beginning at the children
    // cout << "Splitting on left child: size " << root->child1->inputs.size() << endl;
    create_tree(root->child1);
    // cout << "Splitting on right child: size " << root->child2->inputs.size() << endl;
    create_tree(root->child2);
  } else {
    // no better split found
    // cout << "making leaf" << endl;
    make_leaf(root);
  }
}

void MRF::make_leaf(Node* root) {
  // calculate mean_output in the leaf
  root->mean_output = new vector<float>;
  vector<vector<float>* >::iterator it;
  for (int i = 0; i < num_output_vars; i++) {
    float total = 0;
    for (it = root->outputs.begin(); it != root->outputs.end(); it++) {
      total += (*it)->at(i);
    }
    root->mean_output->push_back(total / root->outputs.size());
  }
  root->split_variable = -1;
  root->split_value = -1;
}

bool MRF::perform_best_split(Node* root) {
  // determine random subset of size mtry
  vector<int> subset(mtry, 0);
  int m = mtry; // number left to select
  for (int i = 0; i < num_input_vars; i++) {
    if (rand() % (num_input_vars-i) < m) {
      subset[m-1] = i;
      m--;
    }
  }
  // determine overall sum of each output variable with entities in node  
  vector<float> overall_sum;
  for (int j = 0; j < num_output_vars; j++) {
    float total = 0;
    vector<vector<float>* >::iterator it;
    for (it = root->outputs.begin(); it != root->outputs.end(); it++) {
      total += (*it)->at(j);
    }
    overall_sum.push_back(total);
  }
  int num_node_inputs = root->inputs.size();
  float root_impurity = get_node_impurity_sum(overall_sum, num_node_inputs);
  for (int k = 0; k < overall_sum.size(); k++) {
    root_impurity += pow(overall_sum[k], 2);
  }
  root_impurity = root_impurity / num_node_inputs;
  // float root_impurity = get_node_impurity(root);
  Split best_split = {-1, -1, -1};
  float best_split_score = -1;
  for (int i = 0; i < mtry; i++) {
    // cout << i << " out of " << mtry << endl;
    // determine values of all entities for this input variable
    int variable_index = subset.at(i);
    vector<int> input_indices;
    vector<float> var_values;
    for (int j = 0; j < num_node_inputs; j++) {
      var_values.push_back(root->inputs.at(j)->at(variable_index));
      input_indices.push_back(j);
    }
    // sort entities' indices by these values
    sort(input_indices.begin(), input_indices.end(),
         index_cmp<vector<float>&>(var_values));
    sort(var_values.begin(), var_values.end());
    // iteratively add each to left child and subtract from right, check split score
    vector<float> left_sum(num_output_vars, 0);
    vector<float> right_sum(overall_sum);
    int j; // will be first index in right child
    for (j = -1; j < num_node_inputs; ) {
      float split; // left is <= split, right is > split
      if (j != -1) { // start with empty left child
        do {
          for (int h = 0; h < num_output_vars; h++) {
            left_sum[h] += root->outputs[input_indices[j]]->at(h);
            right_sum[h] -= root->outputs[input_indices[j]]->at(h);
          }
          j++;
          // group together entities with same value to reduce computation
        } while (j != num_node_inputs && var_values.at(j) == var_values.at(j-1));
        split = var_values[j];
      } else {
        j++;
        split = var_values[0] - 1;
      }
      
      float left_impurity = get_node_impurity_sum(left_sum, j);
      int right_size = (j == -1) ? num_node_inputs : num_node_inputs - j;
      float right_impurity = get_node_impurity_sum(right_sum, right_size);
      float split_score = left_impurity + right_impurity - root_impurity;
      Split split_struct = {variable_index, split, j};
      if (split_score > best_split_score) {
        best_split = split_struct;
        best_split_score = split_score;
      }
    }
  }
  // cout << "Best split index: " << best_split.variable_index << endl;
  // cout << "Best split value: " << best_split.split_value << endl;
  // cout << "Best split score: " << best_split_score << endl;
  // cout << "Best split j: " << best_split.input_index << endl;
  if (best_split.input_index <= 0 || best_split.input_index == num_node_inputs) {
    return false;
  }
  // record and execute this split
  // put appropriate entities in each child for best split
  root->child1 = new Node;
  root->child2 = new Node;
  split_node(root, best_split.variable_index, best_split.split_value, root->child1,
             root->child2);
  root->split_variable = best_split.variable_index;
  root->split_value = best_split.split_value;
  return true;
}

float MRF::get_node_impurity_sum(vector<float>& sum, int size) {
  float impurity = 0;
  for (int k = 0; k < sum.size(); k++) {
    impurity += sum[k] * sum[k];
  }
  return (size <= 0) ? 0 : (impurity / size);
}

void MRF::split_node(Node* node, int var_index, float var_value, Node* child1,
                     Node* child2) {
  for (int i = 0; i < node->inputs.size(); i++) {
    vector<float>* input = node->inputs.at(i);
    vector<float>* output = node->outputs.at(i);
    if (input->at(var_index) >= var_value) {
      child2->inputs.push_back(input);
      child2->outputs.push_back(output);
    } else {
      child1->inputs.push_back(input);
      child1->outputs.push_back(output);
    }
  }
}

float MRF::get_node_impurity(Node* node) {
  // find the mean of each variable for the inputs in the node
  vector<float> means;
  vector<vector<float>* >::iterator it;
  for (int i = 0; i < num_output_vars; i++) {
    float total = 0;
    for (it = node->outputs.begin(); it != node->outputs.end(); it++) {
      total += (*it)->at(i);
    }
    means.push_back(total / node->outputs.size());
  }

  // sum the squares of the errors from this mean
  float sum_of_squares = 0;
  for (it = node->outputs.begin(); it != node->outputs.end(); it++) {
    for (int i = 0; i < num_output_vars; i++) {
      sum_of_squares += pow((*it)->at(i) - means.at(i), 2);
    }
  }
  return sum_of_squares;
}

void MRF::output_estimate(vector<float>* input, vector<float>* output,
                          vector<Node*>& trees) {
  vector<Node*>::iterator it;
  for (it = trees.begin(); it != trees.end(); it++) {
    Node* node = *it;
    // determine the leaf in this tree where these inputs would lead
    while (node->child1 != NULL) {
      int index = node->split_variable;
      float split_value = node->split_value;
      if (input->at(index) >= split_value) {
        node = node->child2;
      } else {
        node = node->child1;
      }
    }
    // add this output "vote" to a cumulative sum
    for (int i = 0; i < num_output_vars; i++) {
      output->at(i) += node->mean_output->at(i);
    }
  }
  // divide sum by the number of votes to get average
  vector<float>::iterator it2;
  for (it2 = output->begin(); it2 != output->end(); it2++) {
    *it2 = *it2 / trees.size();
  }
}

void MRF::determine_predictions_errors() {
  determine_predictions();
  prediction_errors.clear();
  for (int i = 0; i < num_inputs; i++) {
    float mse = MSE(all_outputs.at(i), predictions.at(i));
    prediction_errors.push_back(mse);
  }
}

float MRF::MSE(vector<float>* actual, vector<float>* predicted) {
  float square_error = 0;
  for (int i = 0; i < num_output_vars; i++) {
    square_error += pow((actual->at(i) - predicted->at(i)), 2);
  }
  return square_error / num_output_vars;
}

void MRF::determine_predictions() {
  // determine normalized predictions
  vector<vector<float>* >::iterator it;
  for (it = predictions.begin(); it != predictions.end(); it++) {
    if (*it != NULL) {
      delete *it;
    }
  }
  predictions.clear();
  for (int i = 0; i < num_inputs; i++) {
    vector<float>* output = NULL;
    output = new vector<float>(num_output_vars, 0);
    output_estimate(all_inputs->at(i), output, *(OOB_trees_for_inputs.at(i)));
    predictions.push_back(output);
  }
}

void MRF::determine_variable_stats(vector<vector<float>* >* matrix,
                                   vector<float>& means,
                                   vector<float>& deviations) {
  for (int i = 0; i < matrix->at(0)->size(); i++) {
    // determine all the values of this variable
    vector<float> values;
    vector<vector<float>* >::iterator it;
    for (it = matrix->begin(); it != matrix->end();
         it++) {
      values.push_back((*it)->at(i));
    }
    // find the mean
    float mean = calculate_mean(values);
    means.push_back(mean);
    float deviation = calculate_standard_deviation(values);
    deviations.push_back(deviation);
  }
}

float MRF::ranf() {
  return float(rand()) / float(RAND_MAX);
}

float MRF::random_normal(float mean, float dev) {
  float x1, x2, w, y1;
  static float y2;
  static int use_last = 0;

  if (use_last) {
    y1 = y2;
    use_last = 0;
  }
  else {
    do {
      x1 = 2.0 * ranf() - 1.0;
      x2 = 2.0 * ranf() - 1.0;
      w = x1 * x1 + x2 * x2;
    } while (w >= 1.0);

    w = sqrt((-2.0 * log(w)) / w);
    y1 = x1 * w;
    y2 = x2 * w;
    use_last = 1;
  }
  return (mean + y1 * dev);
} 

void MRF::print_trees(const char* filename) {
  ofstream file(filename);
  vector<Node*>::iterator it;
  for (it = roots.begin(); it != roots.end(); it++) {
    print_node(file, *it, true, 0, (*it)->split_variable, (*it)->split_value);
    file << endl;
  }
}

void MRF::print_trees_new(const char* filename) {
  ofstream file(filename);
  for (int i = 0; i < roots.size(); i++) {
    num_nodes = 0;
    vector<int>::iterator it2;
    for (it2 = OOB_inputs_for_trees[i]->begin(); it2 < OOB_inputs_for_trees[i]->end(); it2++) {
      file << *it2;
      if (it2 != OOB_inputs_for_trees[i]->end() - 1) {
        file << " ";
      }
    }
    file << endl;
    print_node_new(file, roots[i]);
    file << endl;
  }
}

void MRF::print_OOB_inputs_for_trees(const char* filename) {
  ofstream file(filename);
  vector<vector<int>* >::iterator it;
  vector<int>::iterator it2;
  for (it = OOB_inputs_for_trees.begin(); it < OOB_inputs_for_trees.end(); it++) {
    vector<int>* OOB_inputs = *it;
    for (it2 = OOB_inputs->begin(); it2 < OOB_inputs->end(); it2++) {
      file << *it2 << " ";
    }
    file << endl;
  }
}

void MRF::print_OOB_trees_for_inputs(const char* filename) {
  ofstream file(filename);
  vector<vector<Node*>* >::iterator it;
  vector<Node*>::iterator it2;
  for (it = OOB_trees_for_inputs.begin(); it < OOB_trees_for_inputs.end(); it++) {
    vector<Node*>* OOB_trees = *it;
    for (it2 = OOB_trees->begin(); it2 < OOB_trees->end(); it2++) {
      file << (*it2)->tree_index << " ";
    }
    file << endl;
  }
}

int MRF::print_node_new(ofstream& file, Node* node) {
  int left = -1;
  int right = -1;
  if (node->child1 != NULL) {
    left = print_node_new(file, node->child1);
    right = print_node_new(file, node->child2);
  }
  file << num_nodes;
  file << " ";
  file << left;
  file << " ";
  file << right;
  file << " ";
  file << node->split_variable;
  file << " ";
  file << node->split_value;
  if (node->mean_output != NULL) {
    file << endl;
    for (int i = 0; i < node->mean_output->size(); i++) {
      file << node->mean_output->at(i);
      if (i != node->mean_output->size() - 1) {
        file << " ";
      }
    }
  }
  file << endl;
  num_nodes += 1;
  return num_nodes - 1;
}

void MRF::print_node(ofstream& file, Node* node, bool left, int level,
                     int split_index, float split_value) {
  for (int i = 0; i < level+1; i++) {
    file << "|\t";
  }
  if (level == 0) {
    file << "root";
  } else {
    string comparison = left ? "<" : ">=";
    file << split_index << comparison << split_value;
  }
  file << ": " << get_node_impurity(node) << " " << node->inputs.size() << endl;
  int index = node->split_variable;
  float value = node->split_value;
}

void MRF::write_predictions_errors(const char* filename_predictions,
                                   const char* filename_errors,
                                   const char* norm_filename_predictions,
                                   const char* norm_filename_errors) {
  if (norm_filename_predictions != NULL) {
    write_output(norm_filename_predictions, predictions);
  }  
  vector<vector<float>* > predictions_unnorm;
  // remove normalization
  for (int i = 0; i < num_inputs; i++) {
    vector<float>* output_unnorm = NULL;
    output_unnorm = new vector<float>(*(predictions[i]));
    for (int j = 0; j < num_output_vars; j++) {
      output_unnorm->at(j) = (output_unnorm->at(j) * output_devs[j]) + 
          output_means[j];
    }
    predictions_unnorm.push_back(output_unnorm);
  }
  write_output(filename_predictions, predictions_unnorm);
  write_MSEs(predictions_unnorm, filename_errors, norm_filename_errors);
  for (int i = 0; i < num_inputs; i++) {
    delete predictions_unnorm[i];
  }
}

void MRF::write_MSEs(vector<vector<float>* >& predictions_unnorm,
                     const char* filename, const char* norm_filename) {
  vector<vector<float>* >  matrix;
  if (norm_filename != NULL) {
    matrix.push_back(&prediction_errors);
    write_output(norm_filename, matrix);
  }
  matrix.clear();
  vector<float> MSEs_unnorm;
  for (int i = 0; i < num_inputs; i++) {
    float mse = MSE(all_outputs_unnorm->at(i), predictions_unnorm.at(i));
    MSEs_unnorm.push_back(mse);
  }
  matrix.push_back(&MSEs_unnorm);
  write_output(filename, matrix);
}

void MRF::write_output(const char* filename, vector<vector<float>* >& matrix) {
  FILE * outfile = fopen(filename, "w");
  // print the output vectors
  /*for (int i = 0; i < num_inputs; i++) {
    if (!used[i]) {
      fprintf(outfile, "%d ", i);
    }
  }
  fprintf(outfile, "\n");*/

  vector<vector<float>* >::iterator it2;
  for (it2 = matrix.begin(); it2 != matrix.end(); it2++) {
    if ((*it2) != NULL) {
      vector<float>::iterator it3;
      for (it3 = (*it2)->begin(); it3 != (*it2)->end(); it3++) {
        fprintf(outfile, "%.6f ", *it3);
      }
      fprintf(outfile, "\n");
    }
  }
  fclose(outfile);
}

void MRF::read_trees(const char* filename) {
  ifstream file(filename);
  string line;
  vector<string> tokens;
  vector<string>::iterator it;
  if (file.is_open()) {
    getline(file, line);
    while (file.good()) {
      // get OOB
      tokenize(line, tokens);
      vector<int>* OOB_inputs = new vector<int>;
      for (it = tokens.begin(); it != tokens.end(); it++) {
        int ind = get_int(*it);
        OOB_inputs->push_back(ind);
      }
      OOB_inputs_for_trees.push_back(OOB_inputs);
      getline(file, line);
      // get nodes
      vector<Node*> nodes;
      while (line.compare("") != 0) {
        Node* node = new Node;
        nodes.push_back(node);
        tokenize(line, tokens);
        if (tokens[1].compare("-1") == 0) {
          // leaf
          node->mean_output = new vector<float>;
          string outputs;
          getline(file, outputs);
          vector<string> output_tokens;
          tokenize(outputs, output_tokens);
          for (it = output_tokens.begin(); it < output_tokens.end(); it++) {
            float f = get_float(*it);
            node->mean_output->push_back(f);
          }
          node->split_variable = -1;
          node->split_value = -1;
        } else {
          int left_child = get_int(tokens[1]);
          int right_child = get_int(tokens[2]);
          node->split_variable = get_int(tokens[3]);
          node->split_value = get_float(tokens[4]);
          node->child1 = nodes[left_child];
          node->child2 = nodes[right_child];
        }
        getline(file, line);
      }
      Node* root = nodes[nodes.size()-1];
      vector<int>::iterator it2;
      for (it2 = OOB_inputs->begin(); it2 != OOB_inputs->end(); it2++) {
        OOB_trees_for_inputs.at(*it2)->push_back(root);
      }
      root->tree_index = roots.size();
      roots.push_back(root);
      if (log_after) {
        // log_tree(root->tree_index);
      }
      getline(file, line);
    }
  } else {
    cout << "Could not open input file: " << filename << endl;
  }
  cout << "Done reading tree file" << endl;
}

float MRF::get_float(string& val) {
  float f = 0;
  istringstream iss(val);
  if (!(iss >> f)) {
    cout << "Invalid data: " << val << endl;
  }
  return f;
}

int MRF::get_int(string& val) {
  int f = 0;
  istringstream iss(val);
  if (!(iss >> f)) {
    cout << "Invalid data: " << val << endl;
  }
  return f;
}

void MRF::read_data(const char* filename, vector<vector<float>* >& matrix,
                    vector<bool>* discrete) {
  string line;
  ifstream file(filename);
  vector<string> tokens;
  vector<string>::iterator it;
  if (file.is_open()) {
    // determine which fields are discrete if applicable
    if (discrete != NULL) {
      getline(file, line);
      tokenize(line, tokens);
      for (it = tokens.begin(); it != tokens.end(); it++) {
        if ((*it).compare("d") == 0) {
          discrete->push_back(true);
        } else {
          discrete->push_back(false);
        }
      }
    }
    // read all lines of data, put into matrix
    while (file.good()) {
      getline(file, line);
      tokenize(line, tokens);
      vector<float>* values = new vector<float>;
      if (tokens.size() == 0) {
        break;
      }
      for (it = tokens.begin(); it != tokens.end(); it++) {
        float f = 0;
        istringstream iss(*it);
        if (!(iss >> f)) {
          cout << "Invalid data: " << *it << endl;
        }
        values->push_back(f);
      }
      matrix.push_back(values);
    }
  } else {
    cout << "Could not open input file: " << filename << endl;
  }
  cout << "Done reading input file. Checking..." << endl;
  // check that data vectors all have the same length
  size_t length = matrix.at(0)->size();
  vector<vector<float>* >::iterator it2;
  for (it2 = matrix.begin(); it2 != matrix.end(); it2++) {
    if ((*it2)->size() != length) {
      cout << "Invalid data length: " << (*it2)->size() << endl;
    }
  }
  if (discrete != NULL) {
    if (discrete->size() != length) {
      cout << "Invalid discrete line length" << endl;
    }
  }
}

void MRF::tokenize(string line, vector<string>& tokens) {
  tokens.clear();
  istringstream iss(line);
  copy(istream_iterator<string>(iss),
       istream_iterator<string>(),
       back_inserter<vector<string> >(tokens));
}

void MRF::print_int_vector(vector<int>& vec) {
  vector<int>::iterator it;
  for (it = vec.begin(); it != vec.end(); it++) {
    cout << " " << *it;
  }
  cout << endl;
}

void MRF::print_float_vector(vector<float>& vec) {
  vector<float>::iterator it;
  for (it = vec.begin(); it != vec.end(); it++) {
    cout << " " << *it;
  }
  cout << endl;
}

/*static PyObject* MRF_test(PyObject* self, PyObject* args) {
  const int* start;
  
  if (!PyArg_ParseTuple(args, "i", &start)) {
    return NULL;
  }
  cilk_for (int i = 0; i < 10; i++) {
  }

  return Py_BuildValue("i", sts);
}*/

int main(int argc, char** argv) {
  const char* input_file;
  char* output_dir;
  const char* actual_file;
  char* read_file = NULL;
  bool log_after_every_tree = false;
  bool permute = false;
  int num_trees = 1000;
  int terminal_node_size = 20;
  for (int i = 1; i < argc; i++) {
    if (i+1 != argc && strcmp(argv[i], "--output_dir") == 0) {
      output_dir = argv[i+1];
      if (output_dir[strlen(argv[i+1])-1] != '/') {
        output_dir = strcat(argv[i+1], "/");
      } else {
        output_dir = argv[i+1];
      }
      i++;
    } else if (i+1 != argc && strcmp(argv[i], "--input_file") == 0) {
      input_file = argv[i+1];
      i++;
    } else if (i+1 != argc && strcmp(argv[i], "--actual_file") == 0) {
      actual_file = argv[i+1];
      i++;
    } else if (i+1 != argc && strcmp(argv[i], "--read_file") == 0) {
      read_file = argv[i+1];
      i++;
    } else if (i+1 != argc && strcmp(argv[i], "--num_trees") == 0) {
      num_trees = atoi(argv[i+1]);
      i++;
    } else if (i+1 != argc && strcmp(argv[i], "--terminal_node_size") == 0) {
      terminal_node_size = atoi(argv[i+1]);
      i++;
    } else if (strcmp(argv[i], "--log_after_every_tree") == 0) {
      log_after_every_tree = true;
    } else if (strcmp(argv[i], "--permute") == 0) {
      permute = true;
    } else {
      cout << "Invalid arguments\n" << endl;
    }
  }
  vector<vector<float>* > all_inputs;
  vector<bool> discrete;
  cout << "Reading input data: " << input_file << endl;
  MRF::read_data(input_file, all_inputs, &discrete);
  vector<vector<float>* > all_outputs;
  cout << "Reading actual data: " << actual_file << endl;
  MRF::read_data(actual_file, all_outputs, NULL);
  // all_inputs.resize(1000); all_outputs.resize(1000);
  // num_ensembles, mtry, min_terminal_size
  srand(time(NULL));
  MRF mrf(&all_inputs,
          &discrete,
          &all_outputs,
          output_dir,
          read_file,
          log_after_every_tree,
          permute,
          num_trees, 
          0, // defaults to sqrt feature number
          terminal_node_size);
  string out_str(output_dir);
  if (log_after_every_tree) {
    cout << "Writing predictions, MSEs, trees in " << output_dir << endl;
    string predicted_file = out_str + "predicted.txt";
    string MSE_file = out_str + "MSE.txt";
    string predicted_norm_file = out_str + "predicted_norm.txt";
    string MSE_norm_file = out_str + "MSE_norm.txt";
    mrf.write_predictions_errors(predicted_file.c_str(),
                                 MSE_file.c_str(),
                                 predicted_norm_file.c_str(),
                                 MSE_norm_file.c_str());
  } else {
    string trees_file = out_str + "trees.txt";
    mrf.print_trees_new(trees_file.c_str());
  }
}
