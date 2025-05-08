#include <omp.h>
#include <iostream>

using namespace std;

int minVal(int arr[], int n){
    int min_ = arr[0];
    #pragma omp parallel for reduction(min: min_)
    for (int i = 1; i < n; i++){
        if(arr[i] < min_) min_ = arr[i];
    }
    return min_;
}

int maxVal(int arr[], int n){
    int max_ = arr[0];
    #pragma omp parallel for reduction(max: max_)
    for (int i = 1; i < n; i++){
        if(arr[i] > max_) max_ = arr[i];
    }
    return max_;
}


int findsum(int arr[], int n){
    int sum = 0;
    #pragma omp parallel for reduction(+: sum)
    for (int i = 0; i < n; i++){
        sum += arr[i];
    }
    return sum;
}

int avg(int arr[], int n){
    return (double)findsum(arr, n)/n;
} 



int main(){
    int arr[] = {12, 1, 3, 4, 10};
    cout <<  "The minValue is: " <<minVal(arr, 5) << endl;
    cout <<  "The max is: " <<minVal(arr, 5) << endl;
    cout <<  "The sum is: " <<findsum(arr, 5) << endl;
    cout <<  "The average: " <<avg(arr, 5) << endl;
    return 0;
}
