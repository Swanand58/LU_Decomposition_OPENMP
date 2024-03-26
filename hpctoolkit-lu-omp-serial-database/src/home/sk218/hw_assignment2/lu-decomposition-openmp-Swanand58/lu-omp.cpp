#include <iostream>
#include <vector>
#include <numa.h>
#include <omp.h>
#include <random>

using namespace std;

int value = 0;

// long fib(int n)
// {
//   if (n < 2) return n;
//   else return fib(n-1) + fib(n-2);
// }

void intialize_matrix(vector<vector<double>>& matrix, unsigned int seed){

  #pragma omp parallel
  {
      std::mt19937_64 engine(seed);
      std::uniform_real_distribution<double> dist(1.0, 4.0);

      int thread_id = omp_get_thread_num();
      engine.discard(thread_id * 1000000);

      #pragma omp for
      for(int i = 0; i < matrix.size(); i++){
            for(int j = 0; j < matrix[i].size(); j++){
                matrix[i][j] = dist(engine); 
            }
        }
   }

}

void PA_LU_Decomposition(vector<vector<double> >& matrix, int n, vector<double> &pi) {

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
        // #pragma omp parallel for reduction(max: max_val) private(i) shared(matrix, k, n) firstprivate(k_prime)
        // #pragma omp parallel for
        for (i = k; i < n; i++) {
                
                // max_val = std::fabs(matrix[i][k]);
                #pragma omp critical
                {
                    if (max_val < std::fabs(matrix[i][k])){ 
                        max_val = std::fabs(matrix[i][k]);
                        k_prime = i;
                    }
                }
        }
        
        if (max_val == 0) {
            cerr << "Error Singular Matrix:" << endl << k << endl;
            return;
        }

        swap(pi[k], pi[k_prime]);

        // swap(matrix[k], matrix[k_prime]);
        for (int i = 0; i < n; i++) {
            swap(matrix[k][i], matrix[k_prime][i]);
        }

        for (int i = 0; i < k - 1 ; i++) {
            swap(L[k][i], L[k_prime][i]);
        }

        
        U[k][k] = matrix[k][k];

        // #pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            L[i][k] = matrix[i][k] / U[k][k];
            U[k][i] = matrix[k][i];
        }

        //int j;

        //#pragma omp parallel for shared(matrix, L, U, n, k) private(j)
        //#pragma omp parallel for collapse(2)
        // #pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                matrix[i][j] = matrix[i][j] - L[i][k] * U[k][j];
            }
        }
    }

    cout << "Permutation vector pi:\n";
    //#pragma omp for
    // for (int i = 0; i < n; ++i) {
    //     cout << pi[i] << " ";
    // }
    // cout << endl;

    cout << "\nLower Triangular matrix (L):\n";
    //#pragma omp for
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         cout << L[i][j] << "  ";
    //     }
    //     cout << endl;
    // }
    
    cout << "Upper Triangular matrix (U):\n";
    //#pragma omp for
    // for (int i = 0; i < n; ++i) {
    //     for (int j = 0; j < n; ++j) {
    //         cout << U[i][j] << "  ";
    //     }
    //     cout << endl;
    // }
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

  std::cout << name << ": " 
            << matrix_size << " " << nworkers
            << std::endl;

  omp_set_num_threads(nworkers);

  std::vector<vector<double> > matrix(matrix_size, vector<double>(matrix_size));
  std::vector<double> permutation(matrix_size);

  intialize_matrix(matrix, 0);
  cout << "Original Matrix \n";
//   for(int i =0; i< matrix_size; i++) {
//         for(int j =0; j < matrix_size; j++) {
//             cout << matrix[i][j] << " ";
//         }
//         cout << "\n";
//     }

    PA_LU_Decomposition(matrix, matrix_size, permutation);

    return 0;
}
