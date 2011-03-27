#include <vector>
#include <math.h>
#include <time.h>
#include <cilkview.h>
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
#include <reducer_max.h>

#include "MRF.h"

#define DIVISIONS 5
#define PERTURBATIONS 10

MRF::MRF(vector<vector<float>* >* inputs,
         vector<bool>* discrete,
         vector<vector<float>* >* outputs,
         bool update=false,
         int iterations=10,
         int number_to_destroy=1,
         int num_ensembles=1000,
         int mtry=0,
         int min_terminal_size=20) {
  this->num_ensembles = num_ensembles;
  this->min_terminal_size = min_terminal_size;
  this->all_inputs = inputs;
  this->discrete = discrete;
  this->all_outputs_unnorm = outputs;
  this->num_inputs = all_inputs->size();
  this->num_input_vars = all_inputs->at(0)->size();
  this->num_output_vars = all_outputs_unnorm->at(0)->size();
  this->update = update;
  this->number_to_destroy = number_to_destroy;
  
  if (mtry == 0) { // if mtry not specified
    this->mtry = sqrt(num_input_vars);
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

  fprintf(stdout,
          "Generating forest with %d inputs, %d input vars, %d output vars\n",
          num_inputs, num_input_vars, num_output_vars);
  if (update) {
    float initial_probability = 1.0 / float(num_input_vars);
    for (int i = 0; i < num_input_vars; i++) {
      feature_distribution.push_back(initial_probability);
    }
    reorder_input_variables();
  }
  if (update) {
    for (int i = 0; i < iterations; i++) {
      // cout << "Feature distribution: ";
      // print_float_vector(feature_distribution);
      printf("Generating forest for iteration %d\n", i);
      generate_forest();
      // if (i != 0)
      //   destroy_worst_trees();
      stringstream out;
      out << i;
      string iteration = out.str();
      determine_predictions_errors();
      string predict_filename = "predict_" + iteration + ".txt";
      string mse_filename = "MSE_" + iteration + ".txt";
      write_predictions_errors(predict_filename.c_str(), mse_filename.c_str());
      cout << "Wrote predictions and errors" << endl;
      // string feature_filename = "feature_dist_" + iteration + ".txt";
      // write_feature_distribution(feature_filename.c_str());
      // cout << "Wrote feature distribution" << endl;
      // update_feature_distribution();
      // cout << "Updated feature distribution" << endl;
    }
  } else {
    fprintf(stdout,
            "Generating forest with %d inputs, %d input vars, %d output vars\n",
            num_inputs, num_input_vars, num_output_vars);
    generate_forest();
  }
  cout << "Generated forest" << endl;
  determine_predictions_errors();
  cout << "Determined predictions and errors" << endl;
}

MRF::~MRF() {
  vector<Node*>::iterator it;
  for (it = roots.begin(); it != roots.end(); it++) {
    delete *it;
  }
  vector<vector<int>* >::iterator it3;
  for (it3 = OOB.begin(); it3 != OOB.end(); it3++) {
    delete *it3;
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

void MRF::generate_forest() {
  // create subsets of the input with replacement, each in its own root node
  for (int i = 0; i < num_ensembles; i++) {
    Node* root = new Node;
    vector<bool> used(num_inputs, false);
    for (int j = 0; j < num_inputs; j++) {
      int index = rand() % num_inputs;
      if (!(used[index])) {
        used[index] = true;
        root->inputs.push_back(all_inputs->at(index));
        root->outputs.push_back(all_outputs.at(index));
      }
    }
    
    // determine the inputs that are OOB for this tree
    vector<int>* OOB_tree = new vector<int>;
    for (int j = 0; j < num_inputs; j++) {
      if (!(used[j])) {
        OOB_tree->push_back(j);
      }
    }
    OOB.push_back(OOB_tree);
    // printf("Ensemble %d: %d inputs OOB\n", i, count);
    roots.push_back(root);
  }

  time_t start, end;
  time(&start);
  int first = roots.size() - num_ensembles;
  cilk_for (int i = first; i < first + num_ensembles; i++) {
    create_tree(roots.at(i));
    cout << "Created tree " << i << endl;
  }
  time(&end);
  printf("%.2lf seconds taken to create trees\n", difftime(end, start));
  cout << "Done creating trees" << endl;
}

void MRF::destroy_worst_trees() {
  vector<float> tree_errors;
  // determine mean MSE on out of bag inputs for each tree
  vector<int> indices(roots.size(), 0);
  for (int i = 0; i < roots.size(); i++) {
    indices.at(i) = i;
    Node* root = roots.at(i);
    if (root->error == -1) {
      vector<int>* OOB_tree = OOB.at(i);
      vector<Node*> this_tree;
      this_tree.push_back(root);
      float total_MSE = 0;
      for (int j = 0; j < OOB_tree->size(); j++) {
        int input_index = OOB_tree->at(j);
        vector<float>* input = all_inputs->at(input_index);
        vector<float> predicted_output(num_output_vars, 0);
        output_estimate(input,  &predicted_output, this_tree);
        total_MSE += MSE(all_outputs.at(input_index), &predicted_output);
      }
      root->error = total_MSE / float(OOB.size());
    } 
    tree_errors.push_back(root->error);
  }
  print_float_vector(tree_errors);
  sort(indices.begin(), indices.end(),
       index_cmp<vector<float>&>(tree_errors));
  reverse(indices.begin(), indices.end());
  vector<int> to_remove;
  print_int_vector(indices);
  for (int i = 0; i < number_to_destroy; i++) {
    int index = indices.at(i);
    to_remove.push_back(indices.at(i));
  }
  sort(to_remove.begin(), to_remove.end());
  for (int i = 0; i < number_to_destroy; i++) {
    delete roots.at(to_remove.at(i) - i);
    roots.erase(roots.begin() + to_remove.at(i) - i);
  }
}

void MRF::create_tree(Node* root) {
  if (root->inputs.size() <= min_terminal_size) {
    // cout << "Reached leaf of size " << root->inputs.size() << endl;
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
    // don't split past min_terminal_size
    return;
  }
  perform_best_split(root);
  // create the tree beginning at the children
  // cout << "Splitting on left child: size " << root->child1->inputs.size() << endl;
  cilk_spawn create_tree(root->child1);
  // cout << "Splitting on right child: size " << root->child2->inputs.size() << endl;
  create_tree(root->child2);
}

void MRF::perform_best_split(Node* root) {
  // determine random subset of size mtry
  vector<int> subset(mtry, 0);
  // if (!update) {
    int m = mtry; // number left to select
    for (int i = 0; i < num_input_vars; i++) {
      if (rand() % (num_input_vars-i) < m) {
        subset[m-1] = i;
        m--;
      }
    }
  // } else {
  //  dist_sample(feature_distribution, ordering, subset);
  // }
  // cout << "Subset selected: ";
  // print_int_vector(subset);
  
  // determine best split on this subset
  float root_score = get_node_impurity(root);
  Split none = {-1, -1};
  cilk::reducer_max_index<Split, float> max_split(none, -1);
  // cout << "Current node impurity: " << root_score << endl;
  cilk_for (int i = 0; i < mtry; i++) {
    int input_index = subset.at(i);
    float max_value = input_max.at(input_index);
    float min_value = input_min.at(input_index);
    float increment;
    if (discrete->at(input_index) && max_value - min_value < DIVISIONS) {
      increment = 1;
    } else {
      increment = (max_value - min_value) / DIVISIONS;
    }
    int trials = ((max_value - min_value) / increment) + 1;
    cilk_for (int j = 0; j < trials; j++) {
      // cout << "Trial index: " << j << endl;
      Node child1, child2;
      float split_value = min_value + j*increment;
      split_node(root, input_index, split_value, &child1, &child2);
      float split_score = root_score - get_node_impurity(&child1) - 
          get_node_impurity(&child2);
      Split split = {input_index, split_value}; 
      max_split.max_of(split, split_score);
    }
  }
  
  // determine best split on all variables
  float best_split_score = max_split.get_value();
  Split best_split = max_split.get_index();
  float best_split_value = best_split.split_value;
  float best_split_index = best_split.variable_index;

  // cout << "Best split index: " << best_split_index << endl;
  // cout << "Best split value: " << best_split_value << endl;
  // cout << "Best split score: " << best_split_score << endl;
  // record and execute this split
  root->child1 = new Node;
  root->child2 = new Node;
  split_node(root, best_split_index, best_split_value, root->child1,
             root->child2);
  root->split_variable = best_split_index;
  root->split_value = best_split_value;
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
    delete *it;
  }
  predictions.clear();
  for (it = all_inputs->begin(); it != all_inputs->end(); it++) {
    vector<float>* output = new vector<float>(num_output_vars, 0);
    if (output == NULL || *it == NULL) {
      cout << output << " " << *it << endl;
    }
    output_estimate(*it, output, roots);
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
    float mean = accumulate(values.begin(), values.end(), 0.0f) / values.size();
    means.push_back(mean);
    vector<float> zero_mean(values);
    transform(zero_mean.begin(), zero_mean.end(), zero_mean.begin(),
              bind2nd(minus<float>(), mean));
    // find the standard deviation
    float deviation = inner_product(zero_mean.begin(), zero_mean.end(),
                                    zero_mean.begin(), 0.0f );
    deviation = sqrt(deviation / (values.size() - 1 ));
    deviations.push_back(deviation);
  }
}
   
void MRF::update_feature_distribution() {
  float** perturbation_errors = new float* [num_inputs];
  for (int j = 0; j < num_inputs; j ++)
    perturbation_errors[j] = new float[num_input_vars];

  cilk_for (int i = 0; i < num_inputs; i++) {
    cilk_for (int j = 0; j < num_input_vars; j++) { 
      // total perturbation errors for input i with noise on feature j 
      float total_MSE = 0;
      for (int n = 0; n < PERTURBATIONS; n++) { 
        vector<float> features = *(all_inputs->at(i));
        // noise up feature j using a normal distribution 
        float noise = random_normal(input_means[j], input_devs[j]);
        features[j] = noise;
        // compute new prediction error for input
        vector<float> predicted_output(num_output_vars, 0);
        output_estimate(&features, &predicted_output, roots); 
        float mse = MSE(all_outputs_unnorm->at(i), &predicted_output);
        total_MSE += mse;
      }
      // store average over all perturbations
      perturbation_errors[i][j] = total_MSE / float(PERTURBATIONS);
      if (total_MSE != total_MSE) {
        cout << "total mse nan error " << i << " " << j << endl;
      }
    }
  }
  cout << "Prediction Errors: ";
  print_float_vector(prediction_errors);
  // compute weighted variable importance measure
  for (int j = 0; j < num_input_vars; j++) {
    float importance = 0;
    for (int i = 0; i < num_inputs; i++) {
      importance += prediction_errors[i]*perturbation_errors[i][j];
    }
    feature_distribution[j] = importance;
  }
  cout << "Feature Distribution: ";
  print_float_vector(feature_distribution);
  // normalize feature distribution to give probabilities
  float total = accumulate(feature_distribution.begin(),
                           feature_distribution.end(), 0.0f);
  cout << "Total: " << total << endl;
  for (int j = 0; j < num_input_vars; j++) {
    feature_distribution[j] /= float(total);
  }
  reorder_input_variables();

  for (int j = 0; j < num_inputs; j++) {
    delete [] perturbation_errors[j];
  }
  delete [] perturbation_errors;
}

void MRF::reorder_input_variables() {
  ordering.clear();
  for (int i = 0; i < num_input_vars; i++) {
    ordering.push_back(i);
  }
  sort(ordering.begin(), ordering.end(),
       index_cmp<vector<float>&>(feature_distribution));
  reverse(ordering.begin(), ordering.end());
  sort(feature_distribution.begin(), feature_distribution.end());
  reverse(feature_distribution.begin(), feature_distribution.end());
}

void MRF::dist_sample(vector<float> p, vector<int> perm, vector<int>& ans) {
  float rT, mass, totalmass;
  int i, j, k, n1;
  int n = p.size();
  int nans = ans.size();
  
  /* Compute the sample */
  totalmass = 1;
  for (i = 0, n1 = n-1; i < nans; i++, n1--) {
    rT = totalmass * ranf();
    mass = 0;
    for (j = 0; j < n1; j++) {
      mass += p[j];
      if (rT <= mass)
        break;
      }
    ans[i] = perm[j];
    totalmass -= p[j];
    for(k = j; k < n1; k++) {
      p[k] = p[k + 1];
      perm[k] = perm[k + 1];
    }
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

void MRF::print_OOB(const char* filename) {
  ofstream file(filename);
  vector<vector<int>* >::iterator it;
  vector<int>::iterator it2;
  for (it = OOB.begin(); it < OOB.end(); it++) {
    vector<int>* OOB_tree = *it;
    for (it2 = OOB_tree->begin(); it2 < OOB_tree->end(); it2++) {
      file << *it2 << " ";
    }
    file << endl;
  }
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
  if (node->child1 != NULL) {
    print_node(file, node->child1, true, level+1, index, value);
    print_node(file, node->child2, false, level+1, index, value);
  }
}

void MRF::write_predictions_errors(const char* filename_predictions,
                                   const char* filename_errors,
                                   const char* norm_filename_predictions,
                                   const char* norm_filename_errors) {
  if (norm_filename_predictions != NULL) {
    write_output(norm_filename_predictions, predictions);
  }  
  vector<vector<float>* > predictions_unnorm;
  vector<vector<float>* >::iterator it;
  // remove normalization
  for (it = predictions.begin(); it != predictions.end(); it++) {
    vector<float>* output_unnorm = new vector<float>(**it);
    for (int i = 0; i < num_output_vars; i++) {
      output_unnorm->at(i) = (output_unnorm->at(i) * output_devs[i]) + 
          output_means[i];
    }
    predictions_unnorm.push_back(output_unnorm);
  }
  write_output(filename_predictions, predictions_unnorm);
  write_MSEs(predictions_unnorm, filename_errors, norm_filename_errors);
  for (it = predictions_unnorm.begin(); it != predictions_unnorm.end(); it++) {
    delete *it;
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

void MRF::write_feature_distribution(const char* filename) {
  vector<vector<float>* > matrix;
  matrix.push_back(&feature_distribution);
  write_output(filename, matrix);
}

void MRF::write_output(const char* filename, vector<vector<float>* >& matrix) {
  FILE * outfile = fopen(filename, "w");
  // print the output vectors
  vector<vector<float>* >::iterator it2;
  for (it2 = matrix.begin(); it2 != matrix.end(); it2++) {
    vector<float>::iterator it3;
    for (it3 = (*it2)->begin(); it3 != (*it2)->end(); it3++) {
      fprintf(outfile, "%.6f ", *it3);
    }
    fprintf(outfile, "\n");
  }
  fclose(outfile);
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

int cilk_main(int argc, char** argv) {
  vector<vector<float>* > all_inputs;
  vector<bool> discrete;
  cout << "Reading input data" << endl;
  MRF::read_data("motifs_beer.txt", all_inputs, &discrete);
  cout << "Read input data" << endl;
  vector<vector<float>* > all_outputs;
  MRF::read_data("expr_beer.txt", all_outputs, NULL);
  cout << "Read output data" << endl;
  // all_inputs.resize(1000); all_outputs.resize(1000);
  // update, iterations, number_to_destroy, num_ensembles, mtry, min_terminal_size
  MRF mrf(&all_inputs, &discrete, &all_outputs, true, 50, 20, 1, 20, 100);
  mrf.write_predictions_errors("predicted_up_down.txt", 
                               "MSE.txt",
                               "predicted_up_down_norm.txt",
                               "MSE_norm.txt");
  cout << "Wrote predictions and MSEs" << endl;
  mrf.print_trees("trees.txt");
  cout << "Wrote trees" << endl;
  mrf.print_OOB("OOB.txt");
  cout << "Wrote OOB" << endl;
  mrf.write_feature_distribution("feature_dist.txt");
  cout << "Wrote features" << endl;
}
