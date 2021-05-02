#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ************************************************************ //
/*
Demo code - neural network with error back-propagation

This is a fixed implementation of a neural network with 1 hidden layer
and an output of L2 number of nodes ranging (0,1).
The current input values are from a file containing 60,000 images
28p x 28p (784px total) as provided by 
https://www.kaggle.com/zalando-research/fashionmnist/data
and comma seperated value (csv) files are used here. The network
is successful at 53% of the time if trained once. With
TRAIN_NUM_OF_TIMES = 10 the success rises to 76%.

obligatory thanks to my professorsand 3blue1brown

compile with gcc nn_bp.c -o nn_bp -lm -O3
*/
// ************************************************************ //

//#define showErrors

// input vector size
#define V 784
// level sizes
#define L1 120
#define L2 10

// number of times to train the network (with the same set)
#define TRAIN_NUM_OF_TIMES 1

// step of weight adjustment
#define a 0.5

// Neuron data
double DL1[L1];
double DL2[L2];

// Neuron outputs
double OL1[L1];
double OL2[L2];

// Synapses' weights
double WL1[L1][V + 1];
double WL2[L2][L1 + 1];
// ******************** function prototypes ******************** //

void activateNN(double *input);
void trainNN(double *input, double *desired);
double sigmoid(double x);
double sigDer(double x);
int isPrediction(int desiredIndex);
void initWeights(void);
// debug functons
void printDoubleVector(double *vec, int len);

// **************************** main *************************** //

int main()
{
    double input[V];
    double desired[L2];

    initWeights();

    // train the network
    for (int k = 0; k < TRAIN_NUM_OF_TIMES; k++) 
    {
        FILE *stream = fopen("./archive/fashion-mnist_train.csv", "r");

        char line[7000];
        float pixelIntensity;
        int classIndex;
        fgets(line, 7000, stream);

        // get each line of the training data
        while (fgets(line, 7000, stream) != NULL)
        {
            char delim[] = ",";
            // extract the first number of the line
            // which is what the image represents
            char *ptr = strtok(line, delim);
            classIndex = atoi(ptr);
            // create the desired vector
            memset(desired, 0, sizeof(desired));
            desired[classIndex] = 1.0;

            // create the input vector
            int i = 0;
            while (ptr != NULL)
            {
                // the value of the grayscaled pixel is normalized
                pixelIntensity = atoi(ptr) / 255.;
                input[i++] = pixelIntensity;

                ptr = strtok(NULL, delim);
            }
            // feedforward the values
            activateNN(input);
            // train the network
            trainNN(input, desired);
        }
        fclose(stream);
    }

    // count hits and misses
    int predictions[2] = {0, 0};

    // test the network (seperate set of images)
    FILE *stream1 = fopen("./archive/fashion-mnist_test.csv", "r");
    char line[7000];
    float pixelIntensity;
    int classIndex;
    fgets(line, 7000, stream1);

    while (fgets(line, 7000, stream1) != NULL)
    {
        char delim[] = ",";
        char *ptr = strtok(line, delim);

        classIndex = atoi(ptr);
        ptr = strtok(NULL, delim);

        int i = 0;
        while (ptr != NULL)
        {
            pixelIntensity = atoi(ptr) / 255.;
            input[i++] = pixelIntensity;

            ptr = strtok(NULL, delim);
        }
        activateNN(input);
        predictions[(isPrediction(classIndex)) ? 0 : 1]++;
    }
    printf("correct: %d\nincorrect: %d\n", predictions[0], predictions[1]);

    return 0;
}

// ************************* functions ************************* //

// feedforward the input and produce an output
// the sigmoid function is used
void activateNN(double *input)
{
    // Level 1 data & outputs
    for (int i = 0; i < L1; i++)
    {
        DL1[i] = 0;
        for (int j = 0; j < V; j++)
        {
            DL1[i] += input[j] * WL1[i][j];
        }
        // Add bias
        DL1[i] += WL1[i][V];
        // Compute the sigmoid
        OL1[i] = sigmoid(DL1[i]);
    }

    // Level 2 data & outputs (same as for L1)
    for (int i = 0; i < L2; i++)
    {
        DL2[i] = 0;
        for (int j = 0; j < L1; j++)
        {
            DL2[i] += OL1[j] * WL2[i][j];
        }
        DL2[i] += WL2[i][L1];
        OL2[i] = sigmoid(DL2[i]);
    }
}

// train the network
void trainNN(double *input, double *desired)
{
#ifdef showErrors
    double error = 0;
    for (int i = 0; i < L2; i++)
    {
        error += (OL2[i] - desired[i]) * (OL2[i] - desired[i]);
    }
    printf("error: %f\n", error);
#endif

    // LAYER 2 //
    // output errors of the 2nd layer
    double E2[L2];
    // weight errors of the 2nd layer (+ bias)
    double WE2[L2][L1 + 1];

    for (int i = 0; i < L2; i++)
    {
        E2[i] = (OL2[i] - desired[i]) * sigDer(OL2[i]);
        for (int j = 0; j < L1; j++)
        {
            WE2[i][j] = E2[i] * OL1[j];
        }
        WE2[i][L1] = E2[i];
    }

    // LAYER 1 //
    double E1[L1];
    double WE1[L1][V + 1];

    for (int i = 0; i < L1; i++)
    {
        E1[i] = 0;
        // calculate error of hidden layer neuron
        for (int j = 0; j < L2; j++)
        {
            E1[i] += WL2[j][i] * E2[j];
        }
        E1[i] *= sigDer(OL1[i]);

        for (int j = 0; j < V; j++)
        {
            WE1[i][j] = E1[i] * input[j];
            // modify L1 weights
            WL1[i][j] -= a * WE1[i][j];
        }
        WE1[i][L1] = E1[i];
        // modify L1 bias
        WL1[i][L1] -= a * WE1[i][L1];
    }

    // modify L2 weights
    for (int i = 0; i < L2; i++)
    {
        for (int j = 0; j < L1; j++)
        {
            WL2[i][j] -= a * WE2[i][j];
        }
        // bias
        WL2[i][L1] -= a * WE2[i][L1];
    }
}

// sigmoid function
double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

// sigmoid derivative
double sigDer(double x)
{
    return x * (1 - x);
}

// randomly initialize the weights in both layers
void initWeights(void)
{
    // Layer 1
    for (int i = 0; i < L1; i++)
    {
        for (int j = 0; j < V; j++)
            WL1[i][j] = (rand() / (double)RAND_MAX) * 4 - 2;

        WL1[i][V] = (rand() / (double)RAND_MAX) - 2;
    }

    // Layer 2
    for (int i = 0; i < L2; i++)
    {
        for (int j = 0; j < L1; j++)
            WL2[i][j] = (rand() / (double)RAND_MAX) * 4 - 2;
        WL2[i][L1] = (rand() / (double)RAND_MAX) - 2;
    }
}

// find the largest output and check if it's the desired one
int isPrediction(int desiredIndex)
{
    double max = OL2[0];
    int max_index = 0;
    for (int i = 1; i < L2; i++)
    {
        if (OL2[i] > max)
        {
            max = OL2[i];
            max_index = i;
        }
    }
    if (max_index == desiredIndex)
        return 1;
    return 0;
}

void printDoubleVector(double *vec, int len)
{
    for (int i = 0; i < len; i++)
        printf("%f\n", vec[i]);
}