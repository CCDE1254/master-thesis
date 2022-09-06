/**
 * 
 */
package master.thesis.neural.network;

import master.thesis.underlying.UnderlyingPrice;

/**
 * @author QuanLiu
 *
 */
public class NeuralNetworkISPut {

	int numberOfSimulations;
	int numberOfTimeSteps;
	int numberOfFirstHiddenLayerNeurons;
	int numberOfSecondHiddenLayerNeurons;
	double[][] randomNumberMatrix;
	double initialStockPrice;
	double riskFreeRate;
	double volatilityTerm;
	
	double[] timeSeries;
	double strike;
	double maturity;
	
	double upperBoundFactorB;
	double upperBoundExponentialDelta1;
	double lowerBoundFactorA;
	double lowerBoundExponentialDelta2;
	

	public NeuralNetworkISPut(int numberOfSimulations, int numberOfTimeSteps, int numberOfFirstHiddenLayerNeurons,
			int numberOfSecondHiddenLayerNeurons, double[][] randomNumberMatrix, double initialStockPrice,
			double riskFreeRate, double volatilityTerm, double[] timeSeries, double strike, double maturity, double upperBoundFactorB,
			double upperBoundExponentialDelta1, double lowerBoundFactorA, double lowerBoundExponentialDelta2) {
		super();
		this.numberOfSimulations = numberOfSimulations;
		this.numberOfTimeSteps = numberOfTimeSteps;
		this.numberOfFirstHiddenLayerNeurons = numberOfFirstHiddenLayerNeurons;
		this.numberOfSecondHiddenLayerNeurons = numberOfSecondHiddenLayerNeurons;
		this.randomNumberMatrix = randomNumberMatrix;
		this.initialStockPrice = initialStockPrice;
		this.riskFreeRate = riskFreeRate;
		this.volatilityTerm = volatilityTerm;
		this.timeSeries = timeSeries;
		this.strike = strike;
		this.maturity = maturity; 
		this.upperBoundFactorB = upperBoundFactorB;
		this.upperBoundExponentialDelta1 = upperBoundExponentialDelta1;
		this.lowerBoundFactorA = lowerBoundFactorA;
		this.lowerBoundExponentialDelta2 = lowerBoundExponentialDelta2;
	}
	
	double[][] multiplyMatrices(double[][] firstMatrix, double[][] secondMatrix) {
	    double[][] result = new double[firstMatrix.length][secondMatrix[0].length];

	    for (int row = 0; row < result.length; row++) {
	        for (int col = 0; col < result[row].length; col++) {
	            result[row][col] = multiplyMatricesCell(firstMatrix, secondMatrix, row, col);
	        }
	    }

	    return result;
	}
	
	double multiplyMatricesCell(double[][] firstMatrix, double[][] secondMatrix, int row, int col) {
	    double cell = 0;
	    for (int i = 0; i < secondMatrix.length; i++) {
	        cell += firstMatrix[row][i] * secondMatrix[i][col];
	    }
	    return cell;
	}
	
	public double[][] getTransformMatrix(double[][] matrix){
		double[][] transferMatrix = new double[matrix[0].length][matrix.length];
		for(int i = 0; i < matrix.length; i++) {
			for(int j = 0; j < matrix[0].length; j++) {
				transferMatrix[j][i] = matrix[i][j];
			}
		}
		return transferMatrix;
	}

	public double[][] initializeWeightMatrix1() {
		double[][] weightMatrix1 = new double[numberOfSimulations*numberOfTimeSteps + 1][numberOfFirstHiddenLayerNeurons];
		for(int i = 0; i < (numberOfSimulations*numberOfTimeSteps + 1); i++) {
			for(int j = 0; j < numberOfFirstHiddenLayerNeurons; j++) {
				weightMatrix1[i][j] = 0.00001;
			}
		}
		return weightMatrix1;
	}
	
	public double[][] initializeWeightMatrix2() {
		double[][] weightMatrix2 = new double[numberOfFirstHiddenLayerNeurons + 1][numberOfSecondHiddenLayerNeurons];
		for(int i = 0; i < numberOfFirstHiddenLayerNeurons + 1; i++) {
			for(int j = 0; j < numberOfSecondHiddenLayerNeurons; j++) {
				weightMatrix2[i][j] = 0.00001;
			}
		}
		return weightMatrix2;
	}
	
	public double[][] initializeWeightMatrix3() {
		double[][] weightMatrix3 = new double[numberOfSecondHiddenLayerNeurons + 1][2];
		for(int i = 0; i < numberOfSecondHiddenLayerNeurons + 1; i++) {
			for(int j = 0; j < 2; j++) {
				weightMatrix3[i][j] = 0.00001;
			}
		}
		weightMatrix3[0][1] = -1.0/2; 
		return weightMatrix3;
	}
	
	public double[] getInputVector() {
		double[] inputVector = new double[numberOfSimulations*numberOfTimeSteps + 1];
		inputVector[0] = 1.0;
		for(int i = 0; i < numberOfTimeSteps; i++) {
			for(int j = 0; j < numberOfSimulations; j++) {
				inputVector[i*numberOfSimulations+j+1] = randomNumberMatrix[i][j];
			}
		}
		return inputVector;
	}

	
	public double[] getHiddenLayer1Output(double[][] weightMatrix1) {
		double[] inputVector = new double[numberOfSimulations*numberOfTimeSteps + 1];
		inputVector[0] = 1.0;
		for(int i = 0; i < numberOfTimeSteps; i++) {
			for(int j = 0; j < numberOfSimulations; j++) {
				inputVector[i*numberOfSimulations+j+1] = randomNumberMatrix[i][j];
			}
		}
		double[] outputs1 = new double[numberOfFirstHiddenLayerNeurons+1];
		outputs1[0] = 1.0;//bias term
		for(int i = 1; i < numberOfFirstHiddenLayerNeurons+1; i++) {
			double sum = 0.0;
			for(int j = 0; j < numberOfSimulations*numberOfTimeSteps + 1; j++) {
				sum += weightMatrix1[j][i-1]*inputVector[j];
			}
			outputs1[i] = Math.abs(sum);
		}
		return outputs1;
	}
	
	public double[] getHiddenLayer1OutputBeforeReLU(double[][] weightMatrix1) {
		double[] inputVector = new double[numberOfSimulations*numberOfTimeSteps + 1];
		inputVector[0] = 1.0;
		for(int i = 0; i < numberOfTimeSteps; i++) {
			for(int j = 0; j < numberOfSimulations; j++) {
				inputVector[i*numberOfSimulations+j+1] = randomNumberMatrix[i][j];
			}
		}
		double[] outputs1 = new double[numberOfFirstHiddenLayerNeurons+1];
		outputs1[0] = 1.0;//bias term
		for(int i = 1; i < numberOfFirstHiddenLayerNeurons+1; i++) {
			double sum = 0.0;
			for(int j = 0; j < numberOfSimulations*numberOfTimeSteps + 1; j++) {
				sum += weightMatrix1[j][i-1]*inputVector[j];
			}
			outputs1[i] = sum;
		}
		return outputs1;
	}
	
	public double[] getHiddenLayer2Output(double[][] weightMatrix1, double[][] weightMatrix2) {
		double[] inputVector =  getHiddenLayer1Output(weightMatrix1);
		double[] outputs2 = new double[numberOfSecondHiddenLayerNeurons+1];
		outputs2[0] = 1.0;
		for(int i = 1; i < numberOfSecondHiddenLayerNeurons+1; i++) {
			double sum = 0.0;
			for(int j = 0; j < numberOfFirstHiddenLayerNeurons + 1; j++) {
				sum += weightMatrix2[j][i-1]*inputVector[j];
			}
			outputs2[i] = Math.max(sum, 0.0);
		}
		return outputs2;
	}
	
	public double[] getOutputLayerOutput(double[][] weightMatrix1, double[][] weightMatrix2, double[][] weightMatrix3) {
		double[] inputVector =  getHiddenLayer2Output(weightMatrix1,weightMatrix2);
		double[] outputs = new double[2];
		for(int i = 0; i < 2; i++) {
			double sum = 0.0;
			for(int j = 0; j < numberOfSecondHiddenLayerNeurons + 1; j++) {
				sum += weightMatrix3[j][i]*inputVector[j];
			}
			outputs[i] = sum;
		}
		return outputs;
	}
	
	public double[] getOutput(double[][] weightMatrix1, double[][] weightMatrix2, double[][] weightMatrix3) {
		double[] inputVector =  getOutputLayerOutput(weightMatrix1, weightMatrix2, weightMatrix3);
		double[] outputs3 = new double[2];
		outputs3[0] = inputVector[0];
		outputs3[1] = Math.min(inputVector[1],0);
		return outputs3;
	}
	
	public double[][] getDerivativeOfWeightMatrix3(double[][] weightMatrix1, double[][] weightMatrix2, double[][] weightMatrix3, double[] output1, double[] output2, double[] output3) {
		double[][] derivativeMatrix = new double[numberOfSecondHiddenLayerNeurons + 1][2];
		
		UnderlyingPrice normalDistributedRandomNumber = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] normalDistributedRandomNumberMatrix = normalDistributedRandomNumber.getNormalDistributedRandomNumberMatrixUnderTargetDistribution();
		
		UnderlyingPrice underlyingPrice = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] underlyingPriceMatrix = underlyingPrice.getUnderlyingPriceMatrix();
		
		
		
		double derivativeLossFunctionEta1 = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			double payoffValid = 1.0;
			double sumOfNormalNumbers = 0.0;
			double sumOfSquaredNormalNumbers = 0.0;
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][j]>=upperBound) || (underlyingPriceMatrix[i][j]<=lowerBound)) {
					payoffValid = 0.0;
				}
				sumOfNormalNumbers += normalDistributedRandomNumberMatrix[i][j];
				sumOfSquaredNormalNumbers += Math.pow(normalDistributedRandomNumberMatrix[i][j], 2.0);
			}
			double value = (Math.exp(-2.0*riskFreeRate*maturity)/numberOfSimulations)*Math.pow(Math.max((strike - underlyingPriceMatrix[timeSeries.length-1][j]),0.0)*payoffValid,2.0)
					*Math.pow(-2.0*output3[1], numberOfTimeSteps/2.0)
					*(Math.exp(numberOfTimeSteps*Math.pow(output3[0], 2.0)/(4.0*output3[1]) - output3[0]*sumOfNormalNumbers - (1.0/2+output3[1])*sumOfSquaredNormalNumbers)*(numberOfTimeSteps*output3[0]/(2.0*output3[1]) - sumOfNormalNumbers) );
			derivativeLossFunctionEta1 += value;
		}
		
		double derivativeLossFunctionEta2 = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			double payoffValid = 1.0;
			double sumOfNormalNumbers = 0.0;
			double sumOfSquaredNormalNumbers = 0.0;
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][j]>=upperBound) || (underlyingPriceMatrix[i][j]<=lowerBound)) {
					payoffValid = 0.0;
				}
				sumOfNormalNumbers += normalDistributedRandomNumberMatrix[i][j];
				sumOfSquaredNormalNumbers += Math.pow(normalDistributedRandomNumberMatrix[i][j], 2.0);
			}
			double value = -(Math.exp(-2.0*riskFreeRate*maturity)*numberOfTimeSteps/numberOfSimulations)*Math.pow(Math.max((strike - underlyingPriceMatrix[timeSeries.length-1][j]),0.0)*payoffValid,2.0)
					*Math.pow(-2.0*output3[1], numberOfTimeSteps/2.0-1.0)
					*(Math.exp(numberOfTimeSteps*Math.pow(output3[0], 2.0)/(4.0*output3[1]) - output3[0]*sumOfNormalNumbers - (1.0/2+output3[1])*sumOfSquaredNormalNumbers))
					+ (Math.exp(-2.0*riskFreeRate*maturity)/numberOfSimulations)*Math.pow(Math.max((strike - underlyingPriceMatrix[timeSeries.length-1][j]),0.0)*payoffValid,2.0)
					*Math.pow(-2.0*output3[1], numberOfTimeSteps/2.0)
					*(Math.exp(numberOfTimeSteps*Math.pow(output3[0], 2.0)/(4.0*output3[1]) - output3[0]*sumOfNormalNumbers - (1.0/2+output3[1])*sumOfSquaredNormalNumbers)*(-numberOfTimeSteps*Math.pow(output3[0],2.0)/(4.0*Math.pow(output3[1],2.0)) - sumOfSquaredNormalNumbers));
			derivativeLossFunctionEta2 += value;
		}
		
		double[] derivativeEta1WeightMatrix3Colume1 = new double[numberOfSecondHiddenLayerNeurons + 1];
		for(int i = 0; i < numberOfSecondHiddenLayerNeurons + 1; i++) {
			derivativeEta1WeightMatrix3Colume1[i] = output2[i];
		}
		
		double[] derivativeEta2WeightMatrix3Colume2 = new double[numberOfSecondHiddenLayerNeurons + 1];
		for(int i = 0; i < numberOfSecondHiddenLayerNeurons + 1; i++) {
			if(output3[1]<0) {
				derivativeEta2WeightMatrix3Colume2[i] = output2[i];
			}
			derivativeEta2WeightMatrix3Colume2[i] = 0.0;
		}
		
		for(int i = 0; i < numberOfSecondHiddenLayerNeurons + 1; i++) {
			for(int j = 0; j < 2; j++) {
				if(j==0) {
					derivativeMatrix[i][j] = derivativeLossFunctionEta1*derivativeEta1WeightMatrix3Colume1[i];
				}
				derivativeMatrix[i][j] = derivativeLossFunctionEta2*derivativeEta2WeightMatrix3Colume2[i];
			}
		}
		return derivativeMatrix;
	}

	public double[][] getDerivativeOfWeightMatrix2(double[][] weightMatrix1, double[][] weightMatrix2, double[][] weightMatrix3, double[] output1, double[] output2, double[] output3) {
		double[][] derivativeMatrix = new double[numberOfFirstHiddenLayerNeurons + 1][numberOfSecondHiddenLayerNeurons];
		
		UnderlyingPrice normalDistributedRandomNumber = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] normalDistributedRandomNumberMatrix = normalDistributedRandomNumber.getNormalDistributedRandomNumberMatrixUnderTargetDistribution();
		
		UnderlyingPrice underlyingPrice = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] underlyingPriceMatrix = underlyingPrice.getUnderlyingPriceMatrix();
		
		
		
		double derivativeLossFunctionEta1 = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			double payoffValid = 1.0;
			double sumOfNormalNumbers = 0.0;
			double sumOfSquaredNormalNumbers = 0.0;
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][j]>=upperBound) || (underlyingPriceMatrix[i][j]<=lowerBound)) {
					payoffValid = 0.0;
				}
				sumOfNormalNumbers += normalDistributedRandomNumberMatrix[i][j];
				sumOfSquaredNormalNumbers += Math.pow(normalDistributedRandomNumberMatrix[i][j], 2.0);
			}
			double value = (Math.exp(-2.0*riskFreeRate*maturity)/numberOfSimulations)*Math.pow(Math.max((strike - underlyingPriceMatrix[timeSeries.length-1][j]),0.0)*payoffValid,2.0)
					*Math.pow(-2.0*output3[1], numberOfTimeSteps/2.0)
					*(Math.exp(numberOfTimeSteps*Math.pow(output3[0], 2.0)/(4.0*output3[1]) - output3[0]*sumOfNormalNumbers - (1.0/2+output3[1])*sumOfSquaredNormalNumbers)*(numberOfTimeSteps*output3[0]/(2.0*output3[1]) - sumOfNormalNumbers) );
			derivativeLossFunctionEta1 += value;
		}
		
		double derivativeLossFunctionEta2 = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			double payoffValid = 1.0;
			double sumOfNormalNumbers = 0.0;
			double sumOfSquaredNormalNumbers = 0.0;
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][j]>=upperBound) || (underlyingPriceMatrix[i][j]<=lowerBound)) {
					payoffValid = 0.0;
				}
				sumOfNormalNumbers += normalDistributedRandomNumberMatrix[i][j];
				sumOfSquaredNormalNumbers += Math.pow(normalDistributedRandomNumberMatrix[i][j], 2.0);
			}
			double value = -(Math.exp(-2.0*riskFreeRate*maturity)*numberOfTimeSteps/numberOfSimulations)*Math.pow(Math.max((strike - underlyingPriceMatrix[timeSeries.length-1][j]),0.0)*payoffValid,2.0)
					*Math.pow(-2.0*output3[1], numberOfTimeSteps/2.0-1.0)
					*(Math.exp(numberOfTimeSteps*Math.pow(output3[0], 2.0)/(4.0*output3[1]) - output3[0]*sumOfNormalNumbers - (1.0/2+output3[1])*sumOfSquaredNormalNumbers))
					+ (Math.exp(-2.0*riskFreeRate*maturity)/numberOfSimulations)*Math.pow(Math.max((strike - underlyingPriceMatrix[timeSeries.length-1][j]),0.0)*payoffValid,2.0)
					*Math.pow(-2.0*output3[1], numberOfTimeSteps/2.0)
					*(Math.exp(numberOfTimeSteps*Math.pow(output3[0], 2.0)/(4.0*output3[1]) - output3[0]*sumOfNormalNumbers - (1.0/2+output3[1])*sumOfSquaredNormalNumbers)*(-numberOfTimeSteps*Math.pow(output3[0],2.0)/(4.0*Math.pow(output3[1],2.0)) - sumOfSquaredNormalNumbers));
			derivativeLossFunctionEta2 += value;
		}
		
		double[] derivativeLossFunctionEta = {derivativeLossFunctionEta1, derivativeLossFunctionEta2};
		double[][] derivativeEtaOutput2 = new double[numberOfSecondHiddenLayerNeurons][2];
		for(int i = 0; i < numberOfSecondHiddenLayerNeurons; i++) {
			for(int j = 0; j < 2; j++) {
				if(j==0) {
					derivativeEtaOutput2[i][j] = weightMatrix3[i+1][j];
				}else {
				if(output3[1]==0) {
					derivativeEtaOutput2[i][j] = 0.0;
				}else {
				derivativeEtaOutput2[i][j] = weightMatrix3[i+1][j];
				}
				}
			}
		}
		double[][] derivativeOutput2WeightMatrix2 = new double[numberOfFirstHiddenLayerNeurons + 1][numberOfSecondHiddenLayerNeurons];
		for(int i = 0; i < numberOfFirstHiddenLayerNeurons + 1; i++) {
			for(int j = 0; j < numberOfSecondHiddenLayerNeurons; j++) {
				if(output2[j+1]==0) {
					derivativeOutput2WeightMatrix2[i][j] = 0.0;
				}else {
					derivativeOutput2WeightMatrix2[i][j] = output1[i];
				}
			}
		}
		
		for(int i = 0; i < numberOfFirstHiddenLayerNeurons + 1; i++) {
			for(int j = 0; j < numberOfSecondHiddenLayerNeurons; j++) {
				derivativeMatrix[i][j] = derivativeOutput2WeightMatrix2[i][j]*(derivativeEtaOutput2[j][0]*derivativeLossFunctionEta[0] + derivativeEtaOutput2[j][1]*derivativeLossFunctionEta[1]);
			}
		}
		return derivativeMatrix;
	}	
	
	public double[][] getDerivativeOfWeightMatrix1(double[][] weightMatrix1, double[][] weightMatrix2, double[][] weightMatrix3, double[] output1, double[] output2, double[] output3) {
		double[][] derivativeMatrix = new double[numberOfSimulations*numberOfTimeSteps + 1][numberOfFirstHiddenLayerNeurons];
		
		UnderlyingPrice normalDistributedRandomNumber = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] normalDistributedRandomNumberMatrix = normalDistributedRandomNumber.getNormalDistributedRandomNumberMatrixUnderTargetDistribution();
		
		UnderlyingPrice underlyingPrice = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] underlyingPriceMatrix = underlyingPrice.getUnderlyingPriceMatrix();
		
		
		
		double derivativeLossFunctionEta1 = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			double payoffValid = 1.0;
			double sumOfNormalNumbers = 0.0;
			double sumOfSquaredNormalNumbers = 0.0;
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][j]>=upperBound) || (underlyingPriceMatrix[i][j]<=lowerBound)) {
					payoffValid = 0.0;
				}
				sumOfNormalNumbers += normalDistributedRandomNumberMatrix[i][j];
				sumOfSquaredNormalNumbers += Math.pow(normalDistributedRandomNumberMatrix[i][j], 2.0);
			}
			double value = (Math.exp(-2.0*riskFreeRate*maturity)/numberOfSimulations)*Math.pow(Math.max((strike - underlyingPriceMatrix[timeSeries.length-1][j]),0.0)*payoffValid,2.0)
					*Math.pow(-2.0*output3[1], numberOfTimeSteps/2.0)
					*(Math.exp(numberOfTimeSteps*Math.pow(output3[0], 2.0)/(4.0*output3[1]) - output3[0]*sumOfNormalNumbers - (1.0/2+output3[1])*sumOfSquaredNormalNumbers)*(numberOfTimeSteps*output3[0]/(2.0*output3[1]) - sumOfNormalNumbers) );
			derivativeLossFunctionEta1 += value;
		}
		
		double derivativeLossFunctionEta2 = 0.0;
		for(int j = 0; j < numberOfSimulations; j++) {
			double payoffValid = 1.0;
			double sumOfNormalNumbers = 0.0;
			double sumOfSquaredNormalNumbers = 0.0;
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][j]>=upperBound) || (underlyingPriceMatrix[i][j]<=lowerBound)) {
					payoffValid = 0.0;
				}
				sumOfNormalNumbers += normalDistributedRandomNumberMatrix[i][j];
				sumOfSquaredNormalNumbers += Math.pow(normalDistributedRandomNumberMatrix[i][j], 2.0);
			}
			double value = -(Math.exp(-2.0*riskFreeRate*maturity)*numberOfTimeSteps/numberOfSimulations)*Math.pow(Math.max((strike - underlyingPriceMatrix[timeSeries.length-1][j]),0.0)*payoffValid,2.0)
					*Math.pow(-2.0*output3[1], numberOfTimeSteps/2.0-1.0)
					*(Math.exp(numberOfTimeSteps*Math.pow(output3[0], 2.0)/(4.0*output3[1]) - output3[0]*sumOfNormalNumbers - (1.0/2+output3[1])*sumOfSquaredNormalNumbers))
					+ (Math.exp(-2.0*riskFreeRate*maturity)/numberOfSimulations)*Math.pow(Math.max((strike - underlyingPriceMatrix[timeSeries.length-1][j]),0.0)*payoffValid,2.0)
					*Math.pow(-2.0*output3[1], numberOfTimeSteps/2.0)
					*(Math.exp(numberOfTimeSteps*Math.pow(output3[0], 2.0)/(4.0*output3[1]) - output3[0]*sumOfNormalNumbers - (1.0/2+output3[1])*sumOfSquaredNormalNumbers)*(-numberOfTimeSteps*Math.pow(output3[0],2.0)/(4.0*Math.pow(output3[1],2.0)) - sumOfSquaredNormalNumbers));
			derivativeLossFunctionEta2 += value;
		}
		
		double[][] derivativeLossFunctionEta = {{derivativeLossFunctionEta1, derivativeLossFunctionEta2}};
		double[][] derivativeEtaOutput2 = new double[numberOfSecondHiddenLayerNeurons][2];
		for(int i = 0; i < numberOfSecondHiddenLayerNeurons; i++) {
			for(int j = 0; j < 2; j++) {
				if(j==0) {
					derivativeEtaOutput2[i][j] = weightMatrix3[i+1][j];
				}else {
				if(output3[1]==0) {
					derivativeEtaOutput2[i][j] = 0.0;
				}else {
				derivativeEtaOutput2[i][j] = weightMatrix3[i+1][j];
				}
				}
			}
		}
		double[][] derivativeOutput2Output1 = new double[numberOfFirstHiddenLayerNeurons][numberOfSecondHiddenLayerNeurons];
		for(int i = 0; i < numberOfFirstHiddenLayerNeurons; i++) {
			for(int j = 0; j < numberOfSecondHiddenLayerNeurons; j++) {
				if(output2[j+1]==0) {
					derivativeOutput2Output1[i][j] = 0.0;
				}else {
					derivativeOutput2Output1[i][j] = weightMatrix2[i][j];
				}
			}
		}
		double[][] derivativeOutput1WeightMatrix1 = new double[numberOfSimulations*numberOfTimeSteps + 1][numberOfFirstHiddenLayerNeurons];
		for(int i = 0; i < numberOfSimulations*numberOfTimeSteps + 1; i++) {
			for(int j = 0; j < numberOfFirstHiddenLayerNeurons; j++) {
				if(getHiddenLayer1OutputBeforeReLU(weightMatrix1)[j+1]>0) {
					derivativeOutput1WeightMatrix1[i][j] = getInputVector()[i];
				}else {
					derivativeOutput1WeightMatrix1[i][j] = -getInputVector()[i];
				}
			
			}
		}
		
		for(int i = 0; i < numberOfSimulations*numberOfTimeSteps + 1; i++) {
			for(int j = 0; j < numberOfFirstHiddenLayerNeurons; j++) {
				derivativeMatrix[i][j] = derivativeOutput1WeightMatrix1[i][j]*multiplyMatrices(multiplyMatrices(derivativeLossFunctionEta, getTransformMatrix(derivativeEtaOutput2)),getTransformMatrix(derivativeOutput2Output1))[0][j];
			}
		}
		return derivativeMatrix;
	}	
	
}
