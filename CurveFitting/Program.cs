Random rand = new Random();
Classifier classifier;

while (true)
{
    Console.Clear();
    Console.WriteLine("Curve Fitting Assignment \n");
    Console.WriteLine("Note: 1000 random data samples are generated with uniform distribution. the target for each sample is calculated using the provided function in the assignment. 15% of the data is seperated for validation and 15% is put aside for testing. \n");

    int dataSize = 1000;
    double validationDataRatio = 0.15;
    double testDataRatio = 0.15;

    var inputs = GenerateInputs(dataSize);
    var targets = GenerateTargets(inputs);

    (var validationInputs, var testInputs, var trainingInputs) = SplitInputs(inputs, validationDataRatio, testDataRatio);
    (var validationTargets, var testTargets, var trainingTargets) = SplitTargets(inputs, targets, validationDataRatio, testDataRatio);

    int layerCount = (int)GetHyperParameterFromUser("Enter layer count (number of hidden layers + 1): ");
    double learningRate = GetHyperParameterFromUser("Enter learning rate: ");
    int numOfNeuronsPerLayer = (int)GetHyperParameterFromUser("Enter number of neurons per layer: ");

    Console.WriteLine("----------------------------------------------------------------");

    classifier = new Classifier(trainingInputs, trainingTargets, validationInputs, validationTargets, layerCount, learningRate, numOfNeuronsPerLayer);
    Console.WriteLine("Training started...");
    classifier.Train();
    Console.WriteLine("Training finished!");
    Console.WriteLine($"Last epoch: {classifier.Epoch}, Validation checks: {classifier.ValidationChecks}, Last MSE: {classifier.LastMSE}");

    Console.WriteLine("----------------------------------------------------------------");
    Console.WriteLine("Test results: (values are rounded)\n");
    PrintTestResults(testInputs, testTargets);

    Console.WriteLine("----------------------------------------------------------------");
    Console.WriteLine();

    Console.Write("press any button to retrain the network: ");
    Console.ReadKey();
}


double[,] GenerateInputs(int dataSize)
{
    double[,] output = new double[dataSize, 2];
    for (int i = 0; i < dataSize; i++)
    {
        double x1 = rand.NextDouble();
        double x2 = rand.NextDouble();
        output[i, 0] = x1;
        output[i, 1] = x2;
    }
    return output;
}
double[] GenerateTargets(double[,] inputs)
{
    var output = new double[inputs.GetLength(0)];
    for (int i = 0; i < output.Length; i++)
    {
        output[i] = Function(inputs[i, 0], inputs[i, 1]);
    }
    return output;
}
double Function(double x1, double x2)
{
    return Math.Sin(2 * Math.PI * x1) * Math.Sin(2 * Math.PI * x2);
}
Tuple<double[,], double[,], double[,]> SplitInputs(double[,] inputs, double validationDataRatio, double testDataRatio)
{
    int totalSampleCount = inputs.GetLength(0);
    int featuresCount = inputs.GetLength(1);

    int validationDataSize = (int)(totalSampleCount * validationDataRatio);
    int testDataSize = (int)(totalSampleCount * testDataRatio);
    int trainingDataSize = totalSampleCount - validationDataSize - testDataSize;

    Tuple<double[,], double[,], double[,]> output = new(new double[validationDataSize, featuresCount], new double[testDataSize, featuresCount], new double[trainingDataSize, featuresCount]);

    for (int i = 0; i < validationDataSize; i++)
    {
        for (int j = 0; j < featuresCount; j++)
        {
            output.Item1[i, j] = inputs[i, j];
        }
    }

    for (int i = 0; i < testDataSize; i++)
    {
        for (int j = 0; j < featuresCount; j++)
        {
            output.Item2[i, j] = inputs[validationDataSize + i, j];
        }
    }

    for (int i = 0; i < trainingDataSize; i++)
    {
        for (int j = 0; j < featuresCount; j++)
        {
            output.Item3[i, j] = inputs[validationDataSize + testDataSize + i, j];
        }
    }

    return output;
}
Tuple<double[], double[], double[]> SplitTargets(double[,] inputs, double[] targets, double validationDataRatio, double testDataRatio)
{
    int totalSampleCount = inputs.GetLength(0);

    int validationDataSize = (int)(totalSampleCount * validationDataRatio);
    int testDataSize = (int)(totalSampleCount * testDataRatio);
    int trainingDataSize = totalSampleCount - validationDataSize - testDataSize;

    Tuple<double[], double[], double[]> output = new(new double[validationDataSize], new double[testDataSize], new double[trainingDataSize]);

    for (int i = 0; i < validationDataSize; i++)
    {
        output.Item1[i] = targets[i];
    }

    for (int i = 0; i < testDataSize; i++)
    {
        output.Item2[i] = targets[validationDataSize + i];
    }

    for (int i = 0; i < trainingDataSize; i++)
    {
        output.Item3[i] = targets[validationDataSize + testDataSize + i];
    }

    return output;
}
double GetHyperParameterFromUser(string message)
{
    bool shouldStop = false;
    double output = 0;
    while (!shouldStop)
    {
        Console.Write(message + " ");
        string? userInput = Console.ReadLine();
        shouldStop = double.TryParse(userInput, out output);
        if (!shouldStop) Console.Write("Invalid input. please try again. ");
    }
    return output;
}
void PrintTestResults(double[,] testInputs, double[] testTargets)
{
    for (int i = 0; i < testInputs.GetLength(0); i++)
    {
        var currentRow = new double[testInputs.GetLength(1)];
        for (int j = 0; j < currentRow.Length; j++)
        {
            currentRow[j] = testInputs[i, j];
        }

        var output = classifier.Test(currentRow);
        var answer = testTargets[i];
        var error = Math.Abs(output - answer);
        Console.WriteLine($"{i + 1}. x1: {currentRow[0].ToString("F2")}, x2: {currentRow[1].ToString("F2")}, output: {output.ToString("F2")}, answer: {answer.ToString("F2")}, error: {error.ToString("F2")}");
    }
}