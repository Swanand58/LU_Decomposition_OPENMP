#include <iostream>
#include <vector>
#include <numa.h>
#include <omp.h>
#include <random>
#include <chrono>

using namespace std::chrono;
using namespace std;

// int value = 0;

// long fib(int n)
// {
//   if (n < 2) return n;
//   else return fib(n-1) + fib(n-2);
// }

vector<vector<double>> matrix_multiplication(const vector<vector<double>> &L,const vector<vector<double>> &U){
    int sizem = L.size();

    vector<vector<double>> product(sizem, vector<double>(sizem, 0.0));

    for(int i = 0; i < sizem; i++){
        for(int j = 0; j < sizem; j++){
            double lu = 0;
            for(int k = 0; k < sizem; k++){
                product[i][j] += L[i][k] * U[k][j];
            }
        }
    }
    
    return product;
}

vector<vector<double>> apply_permutation(const vector<double> &pi, const vector<vector<double>> & matrix) {
    
    vector<vector<double>> PA(matrix.size(), vector<double>(matrix.size()));
    for (int i = 0; i < matrix.size(); i++) {
             PA[i] = matrix[pi[i]];
    }
    return PA;
}

double L21_norm(const vector<vector<double>>& matrix) {
    double norm = 0.0;
    for (int i = 0; i < matrix.size(); i++) {
        double csum = 0.0; //column sum
        for (int j = 0; j < matrix.size(); j++) {
            csum += matrix[j][i] * matrix[j][i];
        }
        norm += sqrt(csum);
    }
    return norm;
}

// Function to compute the residual matrix R = PA - LU and its L2,1 norm
double check_LU_decomposition(const vector<vector<double>>& matrix, const vector<vector<double>>& L, const vector<vector<double>>& U, const vector<double>& pi) {
    // Apply permutation matrix P to A
    vector<vector<double>> PA = apply_permutation(pi, matrix);
    
    // Compute LU
    vector<vector<double>> LU = matrix_multiplication(L, U);
    
    // Compute residual matrix R = PA - LU
    vector<vector<double>> R(matrix.size(), vector<double>(matrix[0].size()));
    for (int i = 0; i < matrix.size(); i++)
        for (int j = 0; j < matrix.size(); j++)
            R[i][j] = PA[i][j] - LU[i][j];
    
    // Compute the L2,1 norm of the residual matrix
    return L21_norm(R);
}


void intialize_matrix(vector<vector<double>>& matrix, unsigned int seed){

    std::mt19937 generator(seed);
    std::uniform_real_distribution<> dist(0.5, 3.0);


    #pragma omp parallel for
    for(int i = 0; i < matrix.size(); i++){
        for(int j = 0; j < matrix.size(); j++){
            matrix[i][j] = dist(generator); 
        }
    }


}

void PA_LU_Decomposition(vector<vector<double> >& matrix, int n, vector<double> &pi, vector<vector<double> >& A, bool l2_norm_flag) {

    auto start = high_resolution_clock::now();

    vector<vector<double> > L(n, vector<double>(n, 0));
    vector<vector<double> > U(n, vector<double>(n, 0));

    pi.resize(n);
    
    for (int i = 0; i < n; i++) {
        pi[i] = i;
    }

    // for (int i = 0; i < n; ++i) {
    //     cout << pi[i] << "\n";
    // }

    for (int i = 0; i < n; i++) {
        L[i][i] = 1.0;
    }

    //Printing Initialized (diagonals 1) Lower Traingular Matrix
    // cout << "Lower Triangular matrix (L) Initial :\n";
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         cout << L[i][j] << " \t";
    //     }
    //     cout << endl;
    // }


    for(int k = 0; k < n; k++){
        double max_val = 0.0;
        int k_prime;
        int i;
        for (i = k; i < n; i++) {
            if (max_val < std::fabs(matrix[i][k])){ 
                max_val = std::fabs(matrix[i][k]);
                k_prime = i;
            }
        }
        
        if (max_val == 0) {
            for(int i = 0; i < n; i++){
                cout << matrix[i][k] << " ";
            }
            cout << "\n";
            cout << "Max Value "<<max_val;
            cerr << "Error Singular Matrix:" << endl << k << endl;
            return;
        }

        swap(pi[k], pi[k_prime]);

        for (int i = 0; i < n; i++) {
            swap(matrix[k][i], matrix[k_prime][i]);
        }

        for (int i = 0; i < k ; i++) {
            swap(L[k][i], L[k_prime][i]);
        }

        
        U[k][k] = matrix[k][k];

        #pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            L[i][k] = matrix[i][k] / U[k][k];
            U[k][i] = matrix[k][i];
        }

        #pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                matrix[i][j] = matrix[i][j] - L[i][k] * U[k][j];
            }
        }
    }


    // Printing Permutation matrix
    cout << "Permutation matrix pi:\n";
    // for (int i = 0; i < n; ++i) {
    //     cout << pi[i] << " ";
    // }
    cout << endl;

    // Printing Computer Lower Triangular matrix
    cout << "\nLower Triangular matrix (L):\n";
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         cout << L[i][j] << "  ";
    //     }
    //     cout << endl;
    // }
    
    // Printing Computer Upper Triangular matrix
    cout << "\nUpper Triangular matrix (U):\n";

    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         cout << U[i][j] << "  ";
    //     }
    //     cout << endl;
    // }

    auto stop = high_resolution_clock::now();   // stopping the timer

    auto duration = duration_cast<seconds>(stop - start);  //calculating the lu running time

    cout << "Time taken by LU Decomposition: " << duration.count() << " seconds" << endl;       //printing time

    //flag to turn on, turn off L2 Norm verification.
    if(l2_norm_flag){
        double l2_norm = check_LU_decomposition(A, L, U, pi);
        cout << "L2 Norm of the matrix : "<< l2_norm;
    }

}

void usage(const char *name)
{
	std::cout << "usage: " << name
                  << " matrix-size nworkers"
                  << std::endl;
 	exit(-1);
}


int main(int argc, char **argv)
{

  const char *name = argv[0];

  if (argc < 3) usage(name);

  int matrix_size = atoi(argv[1]);

  int nworkers = atoi(argv[2]);

  bool l2_norm_flag = false;   //flag to toggle checking l2 norm.

  std::cout << name << ": " 
            << matrix_size << " " << nworkers
            << std::endl;

  omp_set_num_threads(nworkers);

  std::vector<vector<double> > matrix(matrix_size, vector<double>(matrix_size)); //define A matrix
  std::vector<double> permutation(matrix_size);  //define pi matrix

  intialize_matrix(matrix, 12);  //function to randomly initialize 'A' matrix
  cout << "Original Matrix \n";
//   for(int i =0; i< matrix_size; i++) {
//         for(int j =0; j < matrix_size; j++) {
//             cout << matrix[i][j] << " ";
//         }
//         cout << "\n";
//     }

    std::vector<vector<double> > A = matrix; //creating the copy of original matrix for l2 norm calculation

    PA_LU_Decomposition(matrix, matrix_size, permutation, A, l2_norm_flag); // LU decomp calculation function

    return 0;
}
