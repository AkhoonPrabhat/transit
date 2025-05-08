#include <iostream>
#include <omp.h>

using namespace std;

void pBubbleSort(int arr[], int n)
{
#pragma omp parallel
    for (int i = 0; i < n; i++)
    {
#pragma omp for
        for (int j = 1; j < n; j += 2)
        {
            if (arr[j] < arr[j - 1])
            {
                swap(arr[j], arr[j - 1]);
            }
        }

#pragma omp barrier
#pragma omp for
        for (int j = 2; j < n; j += 2)
        {
            if (arr[j] < arr[j - 1])
            {
                swap(arr[j], arr[j - 1]);
            }
        }
    }
}

void merge(int arr[], int lo, int mid, int hi)
{
    int nLeft = mid - lo + 1;
    int nRight = hi - mid;

    int arrLeft[nLeft];
    int arrRight[nRight];

    // copy
    for (int i = 0; i < nLeft; i++)
    {
        arrLeft[i] = arr[lo + i];
    }

    for (int i = 0; i < nRight; i++)
    {
        arrRight[i] = arr[mid + 1 + i];
    }

    int i = 0, j = 0, k = lo;

    while (i < nLeft && j < nRight)
    {
        if (arrLeft[i] <= arrRight[j])
        {
            arr[k] = arrLeft[i];
            i++;
        }
        else
        {
            arr[k] = arrRight[j];
            j++;
        }
        k++;
    }

    while (i < nLeft)
    {
        arr[k] = arrLeft[i];
        i++;
        k++;
    }
    while (j < nRight)
    {
        arr[k] = arrRight[j];
        j++;
        k++;
    }
}

void pMergeSort(int arr[], int lo, int hi)
{

    if (lo < hi)
    {
        int mid = (lo + hi) / 2;
#pragma omp parallel sections
        {
#pragma omp section
            {
                pMergeSort(arr, lo, mid);
            }
#pragma omp section
            {
                pMergeSort(arr, mid + 1, hi);
            }
        }
        merge(arr, lo, mid, hi);
    }
}

int main()
{

    int n = 10;
    int arr[n];
    int brr[n];
    double start_time, end_time;

    for (int i = 0, j = n; i < n; i++, j--)
        arr[i] = j;

    // Parallel time
    start_time = omp_get_wtime();
    pMergeSort(arr, 0, n - 1);
    end_time = omp_get_wtime();
    cout << "Parallel Bubble Sort took : " << end_time - start_time << " seconds.\n";
    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";

    return 0;
}