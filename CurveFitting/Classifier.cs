public class Classifier
{
    public double[,] TrainingInputs { get; private set; }
    public double[] TrainingTargets { get; private set; }
    public double[,] ValidationInputs { get; private set; }
    public double[] ValidationTargets { get; private set; }
    public double[][,] LinkWeights { get; private set; }
    public Neuron[][] Neurons { get; private set; }
    public double[][,] WeightAdjustmentParameters { get; private set; }
    public double[][] DeltaFactors { get; private set; }
    public double LearningRate { get; private set; }
    public double LastMSE { get; private set; }
    public int MaxEpoch { get; private set; } = 1000;
    public int MaxValidationChecks { get; private set; } = 40;
    public int Epoch { get; private set; }
    public int ValidationChecks { get; private set; }

    public Classifier(double[,] trainingInputs, double[] trainingTargets, double[,] validationInputs, double[] validationTargets, int layerCount, double learningRate, int numOfHiddenNeuronsPerLayer)
    {
        TrainingInputs = trainingInputs;
        TrainingTargets = trainingTargets;
        ValidationInputs = validationInputs;
        ValidationTargets = validationTargets;
        LearningRate = learningRate;
        Epoch = 0;

        LinkWeights = new double[layerCount][,];
        WeightAdjustmentParameters = new double[layerCount][,];
        Neurons = new Neuron[layerCount + 1][];
        DeltaFactors = new double[layerCount + 1][];

        // initialize neurons
        for (int i = 0; i < layerCount + 1; i++)
        {
            if (i == 0)
            {
                Neuron[] inputNeurons = new Neuron[TrainingInputs.GetLength(1) + 1];
                for (int j = 0; j < inputNeurons.Length; j++)
                {
                    Neuron newNeuron = new();
                    inputNeurons[j] = newNeuron;

                }
                inputNeurons[inputNeurons.Length - 1].ActivityLevel = 1;
                Neurons[0] = inputNeurons;
            }
            else if (i == layerCount)
            {
                Neuron[] outputNeurons = new Neuron[1];
                outputNeurons[0] = new();
                Neurons[layerCount] = outputNeurons;
            }
            else
            {
                Neuron[] hiddenLayerNeurons = new Neuron[numOfHiddenNeuronsPerLayer];
                for (int j = 0; j < hiddenLayerNeurons.Length; j++)
                {
                    Neuron newNeuron = new();
                    hiddenLayerNeurons[j] = newNeuron;
                }
                hiddenLayerNeurons[hiddenLayerNeurons.Length - 1].ActivityLevel = 1;
                Neurons[i] = hiddenLayerNeurons;
            }
            DeltaFactors[i] = new double[Neurons[i].Length];
        }

        //initialize links
        for (int i = 0; i < layerCount; i++)
        {
            double[,] currentLayerWeights = new double[Neurons[i].Length, Neurons[i + 1].Length];
            WeightAdjustmentParameters[i] = new double[Neurons[i].Length, Neurons[i + 1].Length];

            for (int j = 0; j < Neurons[i].Length; j++)
            {
                for (int k = 0; k < Neurons[i + 1].Length; k++)
                {
                    currentLayerWeights[j, k] = 0;
                }
            }
            LinkWeights[i] = currentLayerWeights;
        }
    }
    public void Train()
    {
        Random rnd = new Random();

        // populate link weights
        for (int i = 0; i < LinkWeights.Length; i++)
        {
            for (int j = 0; j < Neurons[i].Length; j++)
            {
                for (int k = 0; k < Neurons[i + 1].Length - 1; k++)
                {
                    LinkWeights[i][j, k] = rnd.NextDouble() - 0.5;
                }
                if (i == LinkWeights.Length - 1)
                {
                    LinkWeights[LinkWeights.Length - 1][j, 0] = rnd.NextDouble() - 0.5;
                }
            }
        }

        bool shouldStop = false;
        while (!shouldStop)
        {
            Epoch++;
            for (int i = 0; i < TrainingInputs.GetLength(0); i++)
            {
                double[] currentRow = new double[TrainingInputs.GetLength(1)];
                for (int j = 0; j < TrainingInputs.GetLength(1); j++)
                {
                    currentRow[j] = TrainingInputs[i, j];
                }

                FeedForward(currentRow);

                var target = TrainingTargets[i];
                BackPropagate(target);
            }
            shouldStop = CheckStopCondition();
        }
    }
    private bool CheckStopCondition()
    {
        double minError = 1.0e-3;
        double meanSquareError = 0;

        for (int i = 0; i < ValidationInputs.GetLength(0); i++)
        {
            double[] currentRow = new double[ValidationInputs.GetLength(1)];
            for (int j = 0; j < currentRow.Length; j++)
            {
                currentRow[j] = ValidationInputs[i, j];
            }
            FeedForward(currentRow);

            double predictedValue = Neurons[Neurons.Length - 1][0].ActivityLevel;
            double target = ValidationTargets[i];
            double squaredError = Math.Pow(target - predictedValue, 2);

            meanSquareError += squaredError;
        }
        meanSquareError /= ValidationTargets.Length;

        if (Epoch == MaxEpoch || ValidationChecks == MaxValidationChecks || meanSquareError <= minError)
        {
            return true;
        }
        if (meanSquareError > LastMSE)
        {
            ValidationChecks++;
        }
        LastMSE = meanSquareError;
        return false;

    }
    private void FeedForward(double[] currentRow)
    {
        // populate input neurons
        for (int i = 0; i < Neurons[0].Length - 1; i++)
        {
            Neurons[0][i].ActivityLevel = currentRow[i];
        }

        // calculate net input and activity level for each hidden layer neuron
        for (int i = 1; i < Neurons.Length - 1; i++)
        {
            for (int j = 0; j < Neurons[i].Length; j++)
            {
                double netInput = 0;
                for (int k = 0; k < Neurons[i - 1].Length; k++)
                {
                    netInput += Neurons[i - 1][k].ActivityLevel * LinkWeights[i - 1][k, j];
                }
                Neurons[i][j].NetInput = netInput;
                if (j != Neurons[i].Length - 1)
                {
                    Neurons[i][j].ActivityLevel = TransferFunction(netInput);
                }
            }
        }

        // calculate net input and activity level for output neuron
        double outputNetInput = 0;
        for (int j = 0; j < Neurons[Neurons.Length - 2].Length; j++)
        {
            outputNetInput += Neurons[Neurons.Length - 2][j].ActivityLevel * LinkWeights[LinkWeights.Length - 1][j, 0];
        }
        Neurons[Neurons.Length - 1][0].NetInput = outputNetInput;
        Neurons[Neurons.Length - 1][0].ActivityLevel = outputNetInput;
    }
    private void BackPropagate(double target)
    {
        int outputLayerIndex = Neurons.Length - 1;
        Neuron outputNeuron = Neurons[outputLayerIndex][0];
        double errorRate = target - outputNeuron.ActivityLevel;

        double deltaFactor = errorRate;
        DeltaFactors[outputLayerIndex][0] = deltaFactor;

        for (int j = 0; j < Neurons[outputLayerIndex - 1].Length; j++)
        {
            double weightAdjustmentParameter = LearningRate * DeltaFactors[outputLayerIndex][0] * Neurons[outputLayerIndex - 1][j].ActivityLevel;
            WeightAdjustmentParameters[outputLayerIndex - 1][j, 0] = weightAdjustmentParameter;
        }

        for (int i = outputLayerIndex - 1; i > 0; i--)
        {
            for (int j = 0; j < Neurons[i].Length; j++)
            {
                double weightedInputDeltaSum = 0;
                for (int k = 0; k < Neurons[i + 1].Length; k++)
                {
                    weightedInputDeltaSum += DeltaFactors[i + 1][k] * LinkWeights[i][j, k];
                }
                DeltaFactors[i][j] = weightedInputDeltaSum * TransferFunctionDerivative(Neurons[i][j].NetInput);

                for (int k = 0; k < Neurons[i - 1].Length; k++)
                {
                    double weightAdjustmentParameter = LearningRate * DeltaFactors[i][j] * Neurons[i - 1][k].ActivityLevel;
                    WeightAdjustmentParameters[i - 1][k, j] = weightAdjustmentParameter;
                }
            }
        }

        for (int i = LinkWeights.Length - 1; i >= 0; i--)
        {
            for (int j = 0; j < Neurons[i].Length; j++)
            {
                for (int k = 0; k < Neurons[i + 1].Length - 1; k++)
                {
                    LinkWeights[i][j, k] += WeightAdjustmentParameters[i][j, k];
                }
                if (i == LinkWeights.Length - 1)
                {
                    LinkWeights[i][j, 0] += WeightAdjustmentParameters[i][j, 0];
                }
            }
        }
    }
    public double Test(double[] testInputs)
    {
        FeedForward(testInputs);
        return Neurons[Neurons.Length - 1][0].ActivityLevel;
    }
    private static double TransferFunction(double netInput)
    {
        return netInput > 0 ? netInput : 0;
    }
    private static double TransferFunctionDerivative(double netInput)
    {
        return netInput > 0 ? 1 : 0;
    }
}