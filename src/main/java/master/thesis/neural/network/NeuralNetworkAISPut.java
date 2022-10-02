/**
 * 
 */
package master.thesis.neural.network;

import master.thesis.underlying.UnderlyingPrice;

/**
 * @author QuanLiu
 *
 */
public class NeuralNetworkAISPut {
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
	

	public NeuralNetworkAISPut(int numberOfSimulations, int numberOfTimeSteps, int numberOfFirstHiddenLayerNeurons,
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

	public double[][][] initializeWeightMatrix1() {
		double[][][] weightMatrix1 = new double[numberOfTimeSteps-1][numberOfTimeSteps][numberOfFirstHiddenLayerNeurons];//first dimension 1, 2, ..., N-1, second dimension for input 1, x0,..., xN-2
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < numberOfTimeSteps; i++) {
				for(int j = 0; j < numberOfFirstHiddenLayerNeurons; j++) {
					weightMatrix1[k][i][j] = -0.000000001;
				}
			}
		}
		
		return weightMatrix1;
	}
	
	public double[][][] initializeWeightMatrix2() {
		double[][][] weightMatrix2 = new double[numberOfTimeSteps-1][numberOfFirstHiddenLayerNeurons + 1][numberOfSecondHiddenLayerNeurons];//first dimension 1, 2, ..., N-1, second dimension for 1 and neurons in first hidden layer
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < numberOfFirstHiddenLayerNeurons + 1; i++) {
				for(int j = 0; j < numberOfSecondHiddenLayerNeurons; j++) {
					weightMatrix2[k][i][j] = -0.000000001;
				}
			}
		}
		
		return weightMatrix2;
	}
	
	public double[][][] initializeWeightMatrix3(double[] etaUnderIS) {
		double[][][] weightMatrix3 = new double[numberOfTimeSteps-1][numberOfSecondHiddenLayerNeurons + 1][2];//first dimension 1, 2, ..., N-1, second dimension for 1 and neurons in second hidden layer
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < numberOfSecondHiddenLayerNeurons + 1; i++) {
				for(int j = 0; j < 2; j++) {
					weightMatrix3[k][i][j] = -0.000000001;
				}
			}
			weightMatrix3[k][0][0] = etaUnderIS[0];
			weightMatrix3[k][0][1] = etaUnderIS[1];
		}
		return weightMatrix3;
	}
	
	
	public double[][] getHiddenLayer1Output(double[][] inputVector, double[][][] weightMatrix1) {//inputVector with size [numberOfTimeSteps-1][numberOfTimeSteps]
		double[][] outputs1 = new double[numberOfTimeSteps-1][numberOfFirstHiddenLayerNeurons+1];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			outputs1[k][0] = 1.0;//bias term
			for(int i = 1; i < numberOfFirstHiddenLayerNeurons+1; i++) {
				double sum = 0.0;
				for(int j = 0; j < k + 2; j++) {
					sum += inputVector[k][j]*weightMatrix1[k][j][i-1];
				}
				outputs1[k][i] = Math.abs(sum);
			}
		}
		
		return outputs1;
	}
	
	public double[][] getHiddenLayer1OutputBeforeReLU(double[][] inputVector, double[][][] weightMatrix1) {
		//inputVector with size [numberOfTimeSteps-1][numberOfTimeSteps], weight matrix 1 with size [numberOfTimeSteps-1][numberOfTimeSteps][numberOfFirstHiddenLayerNeurons]
		double[][] outputs1 = new double[numberOfTimeSteps-1][numberOfFirstHiddenLayerNeurons+1];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			outputs1[k][0] = 1.0;//bias term
			for(int i = 1; i < numberOfFirstHiddenLayerNeurons+1; i++) {
				double sum = 0.0;
				for(int j = 0; j < k + 2; j++) {
					sum += inputVector[k][j]*weightMatrix1[k][j][i-1];
				}
				outputs1[k][i] = sum;
			}
		}
		
		return outputs1;
	}
	
	public double[][] getHiddenLayer2Output(double[][] inputVector, double[][][] weightMatrix1, double[][][] weightMatrix2) {
		/*
		 * inputVector with size [numberOfTimeSteps-1][numberOfTimeSteps], weight matrix 1 with size [numberOfTimeSteps-1][numberOfTimeSteps][numberOfFirstHiddenLayerNeurons], 
		 * weight matrix 2 with size [numberOfTimeSteps-1][numberOfFirstHiddenLayerNeurons + 1][numberOfSecondHiddenLayerNeurons]
		 */
		double[][] inputs =  getHiddenLayer1Output(inputVector, weightMatrix1);
		//input with size [numberOfTimeSteps-1][numberOfFirstHiddenLayerNeurons+1]
		double[][] outputs2 = new double[numberOfTimeSteps-1][numberOfSecondHiddenLayerNeurons+1];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			outputs2[k][0] = 1.0;
			for(int i = 1; i < numberOfSecondHiddenLayerNeurons+1; i++) {
				double sum = 0.0;
				for(int j = 0; j < numberOfFirstHiddenLayerNeurons + 1; j++) {
					sum += inputs[k][j]*weightMatrix2[k][j][i-1];
				}
				outputs2[k][i] = Math.max(sum, 0.0);
			}
		}
		return outputs2;
	}
	
	public double[][] getOutputLayerOutput(double[][] inputVector, double[][][] weightMatrix1, double[][][] weightMatrix2, double[][][] weightMatrix3) {
		/*
		 * inputVector with size [numberOfTimeSteps-1][numberOfTimeSteps], weight matrix 1 with size [numberOfTimeSteps-1][numberOfTimeSteps][numberOfFirstHiddenLayerNeurons], 
		 * weight matrix 2 with size [numberOfTimeSteps-1][numberOfFirstHiddenLayerNeurons + 1][numberOfSecondHiddenLayerNeurons]
		 * weight matrix 3 with size [numberOfTimeSteps-1][numberOfSecondHiddenLayerNeurons + 1][2]
		 */
		double[][] inputs =  getHiddenLayer2Output(inputVector, weightMatrix1,weightMatrix2);
		//input with size [numberOfTimeSteps-1][numberOfSecondHiddenLayerNeurons+1]
		double[][] outputs = new double[numberOfTimeSteps-1][2];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < 2; i++) {
				double sum = 0.0;
				for(int j = 0; j < numberOfSecondHiddenLayerNeurons + 1; j++) {
					sum += weightMatrix3[k][j][i]*inputs[k][j];
				}
				outputs[k][i] = sum;
			}
		}
		return outputs;
	}
	
	public double[][] getOutput(double[][] inputVector, double[][][] weightMatrix1, double[][][] weightMatrix2, double[][][] weightMatrix3) {
		double[][] inputs =  getOutputLayerOutput(inputVector, weightMatrix1, weightMatrix2, weightMatrix3);
		double[][] outputs3 = new double[numberOfTimeSteps-1][2];//first dimension 1, 2, ..., N-1
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			outputs3[k][0] = inputs[k][0];
			outputs3[k][1] = Math.min(inputs[k][1],0);
		}
		return outputs3;
	}
	
	public double[][][] getDerivativeOfWeightMatrix3(double[] etaUnderIS, int iterationStep, double[][][] weightMatrix1, double[][][] weightMatrix2, double[][][] weightMatrix3, double[][] output1, double[][] output2, double[][] output3) {
		//iteration step means that which simulation samples to use in SGD, first iteration use simulation of omega1
		double[][][] derivativeMatrix = new double[numberOfTimeSteps-1][numberOfSecondHiddenLayerNeurons + 1][2];
		
		UnderlyingPrice normalDistributedRandomNumber = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] normalDistributedRandomNumberMatrix = normalDistributedRandomNumber.getNormalDistributedRandomNumberMatrixUnderTargetDistribution();
		
		UnderlyingPrice underlyingPrice = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] underlyingPriceMatrix = underlyingPrice.getUnderlyingPriceMatrix();
		
		double[] derivativeLossFunctionEta1 = new double[numberOfTimeSteps-1];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			double payoffValid = 1.0;
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][iterationStep]>=upperBound) || (underlyingPriceMatrix[i][iterationStep]<=lowerBound)) {
					payoffValid = 0.0;
				}
			}
			double productOfDensity = 1.0;
			productOfDensity *= Math.pow(-2.0*etaUnderIS[1], 1.0/2.0)*Math.exp(Math.pow(etaUnderIS[0],2.0)/(4.0*etaUnderIS[1])
					- etaUnderIS[0]*normalDistributedRandomNumberMatrix[0][iterationStep] - (1.0/2.0 + etaUnderIS[1])*Math.pow(normalDistributedRandomNumberMatrix[0][iterationStep], 2.0));
			for(int i = 0; i < numberOfTimeSteps - 1; i++) {
				productOfDensity *= Math.pow(-2.0*output3[i][1], 1.0/2.0)*Math.exp(Math.pow(output3[i][0],2.0)/(4.0*output3[i][1])
						- output3[i][0]*normalDistributedRandomNumberMatrix[i+1][iterationStep] - (1.0/2.0 + output3[i][1])*Math.pow(normalDistributedRandomNumberMatrix[i+1][iterationStep], 2.0));
			}
			
			derivativeLossFunctionEta1[k] = (Math.exp(-2.0*riskFreeRate*maturity))*Math.pow(Math.max((-underlyingPriceMatrix[timeSeries.length-1][iterationStep] + strike),0.0)*payoffValid,2.0)
					*productOfDensity
					*(output3[k][0]/(2.0*output3[k][1]) - normalDistributedRandomNumberMatrix[k+1][iterationStep]);
	
				
		}
		
	
		
		double[] derivativeLossFunctionEta2 = new double[numberOfTimeSteps-1];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			double payoffValid = 1.0;
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][iterationStep]>=upperBound) || (underlyingPriceMatrix[i][iterationStep]<=lowerBound)) {
					payoffValid = 0.0;
				}
			}
			double productOfDensity = 1.0;
			productOfDensity *= Math.pow(-2.0*etaUnderIS[1], 1.0/2.0)*Math.exp(Math.pow(etaUnderIS[0],2.0)/(4.0*etaUnderIS[1])
					- etaUnderIS[0]*normalDistributedRandomNumberMatrix[0][iterationStep] - (1.0/2.0 + etaUnderIS[1])*Math.pow(normalDistributedRandomNumberMatrix[0][iterationStep], 2.0));
			for(int i = 0; i < numberOfTimeSteps - 1; i++) {
				productOfDensity *= Math.pow(-2.0*output3[i][1], 1.0/2.0)*Math.exp(Math.pow(output3[i][0],2.0)/(4.0*output3[i][1])
						- output3[i][0]*normalDistributedRandomNumberMatrix[i+1][iterationStep] - (1.0/2.0 + output3[i][1])*Math.pow(normalDistributedRandomNumberMatrix[i+1][iterationStep], 2.0));
			}
			double productOfDensity1 = productOfDensity * Math.pow(-2.0*output3[k][1], -1.0);
			double productOfDensity2 = productOfDensity;
			derivativeLossFunctionEta2[k] = -2.0*(Math.exp(-2.0*riskFreeRate*maturity))*Math.pow(Math.max((-underlyingPriceMatrix[timeSeries.length-1][iterationStep] + strike),0.0)*payoffValid,2.0)
					*productOfDensity1
					- (Math.exp(-2.0*riskFreeRate*maturity))*Math.pow(Math.max((-underlyingPriceMatrix[timeSeries.length-1][iterationStep] + strike),0.0)*payoffValid,2.0)
					*productOfDensity2
					*(Math.pow(output3[k][0],2.0)/(4.0*Math.pow(output3[k][1],2.0)) + normalDistributedRandomNumberMatrix[k+1][iterationStep]*normalDistributedRandomNumberMatrix[k+1][iterationStep]);
		}
		
		
		
		double[][] derivativeEta1WeightMatrix3Colume1 = new double[numberOfTimeSteps-1][numberOfSecondHiddenLayerNeurons + 1];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < numberOfSecondHiddenLayerNeurons + 1; i++) {
				derivativeEta1WeightMatrix3Colume1[k][i] = output2[k][i];
			}
		}
		
		
		double[][] derivativeEta2WeightMatrix3Colume2 = new double[numberOfTimeSteps-1][numberOfSecondHiddenLayerNeurons + 1];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < numberOfSecondHiddenLayerNeurons + 1; i++) {
			if(output3[k][1]<0) {
				derivativeEta2WeightMatrix3Colume2[k][i] = output2[k][i];
			    }else {
			    	derivativeEta2WeightMatrix3Colume2[k][i] = 0.0;
			    }
		    }
		}
		
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < numberOfSecondHiddenLayerNeurons + 1; i++) {
			for(int j = 0; j < 2; j++) {
				if(j==0) {
					derivativeMatrix[k][i][j] = derivativeLossFunctionEta1[k]*derivativeEta1WeightMatrix3Colume1[k][i];
				}else {
					derivativeMatrix[k][i][j] = derivativeLossFunctionEta2[k]*derivativeEta2WeightMatrix3Colume2[k][i];
				}
		    }
		  }
		}
		
		return derivativeMatrix;
	}

	public double[][][] getDerivativeOfWeightMatrix2(double[] etaUnderIS,int iterationStep, double[][][] weightMatrix1, double[][][] weightMatrix2, double[][][] weightMatrix3, double[][] output1, double[][] output2, double[][] output3) {
		double[][][] derivativeMatrix = new double[numberOfTimeSteps-1][numberOfFirstHiddenLayerNeurons + 1][numberOfSecondHiddenLayerNeurons];
		
		UnderlyingPrice normalDistributedRandomNumber = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] normalDistributedRandomNumberMatrix = normalDistributedRandomNumber.getNormalDistributedRandomNumberMatrixUnderTargetDistribution();
		
		UnderlyingPrice underlyingPrice = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] underlyingPriceMatrix = underlyingPrice.getUnderlyingPriceMatrix();
		
		
		
		double[] derivativeLossFunctionEta1 = new double[numberOfTimeSteps-1];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			double payoffValid = 1.0;
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][iterationStep]>=upperBound) || (underlyingPriceMatrix[i][iterationStep]<=lowerBound)) {
					payoffValid = 0.0;
				}
			}
			double productOfDensity = 1.0;
			productOfDensity *= Math.pow(-2.0*etaUnderIS[1], 1.0/2.0)*Math.exp(Math.pow(etaUnderIS[0],2.0)/(4.0*etaUnderIS[1])
					- etaUnderIS[0]*normalDistributedRandomNumberMatrix[0][iterationStep] - (1.0/2.0 + etaUnderIS[1])*Math.pow(normalDistributedRandomNumberMatrix[0][iterationStep], 2.0));
			for(int i = 0; i < numberOfTimeSteps - 1; i++) {
				productOfDensity *= Math.pow(-2.0*output3[i][1], 1.0/2.0)*Math.exp(Math.pow(output3[i][0],2.0)/(4.0*output3[i][1])
						- output3[i][0]*normalDistributedRandomNumberMatrix[i+1][iterationStep] - (1.0/2.0 + output3[i][1])*Math.pow(normalDistributedRandomNumberMatrix[i+1][iterationStep], 2.0));
			}
			
			derivativeLossFunctionEta1[k] = (Math.exp(-2.0*riskFreeRate*maturity))*Math.pow(Math.max((-underlyingPriceMatrix[timeSeries.length-1][iterationStep] + strike),0.0)*payoffValid,2.0)
					*productOfDensity
					*(output3[k][0]/(2.0*output3[k][1]) - normalDistributedRandomNumberMatrix[k+1][iterationStep]);
	
				
		}
		
	
		
		double[] derivativeLossFunctionEta2 = new double[numberOfTimeSteps-1];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			double payoffValid = 1.0;
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][iterationStep]>=upperBound) || (underlyingPriceMatrix[i][iterationStep]<=lowerBound)) {
					payoffValid = 0.0;
				}
			}
			double productOfDensity = 1.0;
			productOfDensity *= Math.pow(-2.0*etaUnderIS[1], 1.0/2.0)*Math.exp(Math.pow(etaUnderIS[0],2.0)/(4.0*etaUnderIS[1])
					- etaUnderIS[0]*normalDistributedRandomNumberMatrix[0][iterationStep] - (1.0/2.0 + etaUnderIS[1])*Math.pow(normalDistributedRandomNumberMatrix[0][iterationStep], 2.0));
			for(int i = 0; i < numberOfTimeSteps - 1; i++) {
				productOfDensity *= Math.pow(-2.0*output3[i][1], 1.0/2.0)*Math.exp(Math.pow(output3[i][0],2.0)/(4.0*output3[i][1])
						- output3[i][0]*normalDistributedRandomNumberMatrix[i+1][iterationStep] - (1.0/2.0 + output3[i][1])*Math.pow(normalDistributedRandomNumberMatrix[i+1][iterationStep], 2.0));
			}
			double productOfDensity1 = productOfDensity * Math.pow(-2.0*output3[k][1], -1.0);
			double productOfDensity2 = productOfDensity;
			derivativeLossFunctionEta2[k] = -2.0*(Math.exp(-2.0*riskFreeRate*maturity))*Math.pow(Math.max((-underlyingPriceMatrix[timeSeries.length-1][iterationStep] + strike),0.0)*payoffValid,2.0)
					*productOfDensity1
					- (Math.exp(-2.0*riskFreeRate*maturity))*Math.pow(Math.max((-underlyingPriceMatrix[timeSeries.length-1][iterationStep] + strike),0.0)*payoffValid,2.0)
					*productOfDensity2
					*(Math.pow(output3[k][0],2.0)/(4.0*Math.pow(output3[k][1],2.0)) + normalDistributedRandomNumberMatrix[k+1][iterationStep]*normalDistributedRandomNumberMatrix[k+1][iterationStep]);
		}
		
		double[][] derivativeLossFunctionEta = new double[numberOfTimeSteps - 1][2];
		for(int i = 0; i < numberOfTimeSteps - 1; i++) {
			for(int j = 0; j < 2; j++) {
				if(j==0) {
					derivativeLossFunctionEta[i][j] = derivativeLossFunctionEta1[i];
				}else {
					derivativeLossFunctionEta[i][j] = derivativeLossFunctionEta2[i];
				}
				
			}
		}
		double[][][] derivativeEtaOutput2 = new double[numberOfTimeSteps - 1][numberOfSecondHiddenLayerNeurons][2];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < numberOfSecondHiddenLayerNeurons; i++) {
				for(int j = 0; j < 2; j++) {
					if(j==0) {
						derivativeEtaOutput2[k][i][j] = weightMatrix3[k][i+1][j];
					}else {
					if(output3[k][1]==0) {
						derivativeEtaOutput2[k][i][j] = 0.0;
					}else {
					derivativeEtaOutput2[k][i][j] = weightMatrix3[k][i+1][j];
					}
					}
				}
			}
		}
		
		double[][][] derivativeOutput2WeightMatrix2 = new double[numberOfTimeSteps - 1][numberOfFirstHiddenLayerNeurons + 1][numberOfSecondHiddenLayerNeurons];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < numberOfFirstHiddenLayerNeurons + 1; i++) {
				for(int j = 0; j < numberOfSecondHiddenLayerNeurons; j++) {
					if(output2[k][j+1]==0) {
						derivativeOutput2WeightMatrix2[k][i][j] = 0.0;
					}else {
						derivativeOutput2WeightMatrix2[k][i][j] = output1[k][i];
					}
				}
			}
		}
		
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < numberOfFirstHiddenLayerNeurons + 1; i++) {
				for(int j = 0; j < numberOfSecondHiddenLayerNeurons; j++) {
					derivativeMatrix[k][i][j] = derivativeOutput2WeightMatrix2[k][i][j]*(derivativeEtaOutput2[k][j][0]*derivativeLossFunctionEta[k][0] + derivativeEtaOutput2[k][j][1]*derivativeLossFunctionEta[k][1]);
				}
			}
		}
		
		return derivativeMatrix;
	}	
	
	public double[][][] getDerivativeOfWeightMatrix1(double[] etaUnderIS, int iterationStep, double[][] inputVector, double[][][] weightMatrix1, double[][][] weightMatrix2, double[][][] weightMatrix3, double[][] output1, double[][] output2, double[][] output3) {
		double[][][] derivativeMatrix = new double[numberOfTimeSteps-1][numberOfSimulations][numberOfFirstHiddenLayerNeurons];
		
		UnderlyingPrice normalDistributedRandomNumber = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] normalDistributedRandomNumberMatrix = normalDistributedRandomNumber.getNormalDistributedRandomNumberMatrixUnderTargetDistribution();
		
		UnderlyingPrice underlyingPrice = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] underlyingPriceMatrix = underlyingPrice.getUnderlyingPriceMatrix();
		
		
		
		double[] derivativeLossFunctionEta1 = new double[numberOfTimeSteps-1];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			double payoffValid = 1.0;
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][iterationStep]>=upperBound) || (underlyingPriceMatrix[i][iterationStep]<=lowerBound)) {
					payoffValid = 0.0;
				}
			}
			double productOfDensity = 1.0;
			productOfDensity *= Math.pow(-2.0*etaUnderIS[1], 1.0/2.0)*Math.exp(Math.pow(etaUnderIS[0],2.0)/(4.0*etaUnderIS[1])
					- etaUnderIS[0]*normalDistributedRandomNumberMatrix[0][iterationStep] - (1.0/2.0 + etaUnderIS[1])*Math.pow(normalDistributedRandomNumberMatrix[0][iterationStep], 2.0));
			for(int i = 0; i < numberOfTimeSteps - 1; i++) {
				productOfDensity *= Math.pow(-2.0*output3[i][1], 1.0/2.0)*Math.exp(Math.pow(output3[i][0],2.0)/(4.0*output3[i][1])
						- output3[i][0]*normalDistributedRandomNumberMatrix[i+1][iterationStep] - (1.0/2.0 + output3[i][1])*Math.pow(normalDistributedRandomNumberMatrix[i+1][iterationStep], 2.0));
			}
			
			derivativeLossFunctionEta1[k] = (Math.exp(-2.0*riskFreeRate*maturity))*Math.pow(Math.max((-underlyingPriceMatrix[timeSeries.length-1][iterationStep] + strike),0.0)*payoffValid,2.0)
					*productOfDensity
					*(output3[k][0]/(2.0*output3[k][1]) - normalDistributedRandomNumberMatrix[k+1][iterationStep]);
	
				
		}
		
	
		
		double[] derivativeLossFunctionEta2 = new double[numberOfTimeSteps-1];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			double payoffValid = 1.0;
			for(int i = 0; i < timeSeries.length; i++) {
				double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
				double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
				if((underlyingPriceMatrix[i][iterationStep]>=upperBound) || (underlyingPriceMatrix[i][iterationStep]<=lowerBound)) {
					payoffValid = 0.0;
				}
			}
			double productOfDensity = 1.0;
			productOfDensity *= Math.pow(-2.0*etaUnderIS[1], 1.0/2.0)*Math.exp(Math.pow(etaUnderIS[0],2.0)/(4.0*etaUnderIS[1])
					- etaUnderIS[0]*normalDistributedRandomNumberMatrix[0][iterationStep] - (1.0/2.0 + etaUnderIS[1])*Math.pow(normalDistributedRandomNumberMatrix[0][iterationStep], 2.0));
			for(int i = 0; i < numberOfTimeSteps - 1; i++) {
				productOfDensity *= Math.pow(-2.0*output3[i][1], 1.0/2.0)*Math.exp(Math.pow(output3[i][0],2.0)/(4.0*output3[i][1])
						- output3[i][0]*normalDistributedRandomNumberMatrix[i+1][iterationStep] - (1.0/2.0 + output3[i][1])*Math.pow(normalDistributedRandomNumberMatrix[i+1][iterationStep], 2.0));
			}
			double productOfDensity1 = productOfDensity * Math.pow(-2.0*output3[k][1], -1.0);
			double productOfDensity2 = productOfDensity;
			derivativeLossFunctionEta2[k] = -2.0*(Math.exp(-2.0*riskFreeRate*maturity))*Math.pow(Math.max((-underlyingPriceMatrix[timeSeries.length-1][iterationStep] + strike),0.0)*payoffValid,2.0)
					*productOfDensity1
					- (Math.exp(-2.0*riskFreeRate*maturity))*Math.pow(Math.max((-underlyingPriceMatrix[timeSeries.length-1][iterationStep] + strike),0.0)*payoffValid,2.0)
					*productOfDensity2
					*(Math.pow(output3[k][0],2.0)/(4.0*Math.pow(output3[k][1],2.0)) + normalDistributedRandomNumberMatrix[k+1][iterationStep]*normalDistributedRandomNumberMatrix[k+1][iterationStep]);
		}
		
		double[][][] derivativeLossFunctionEta = new double[numberOfTimeSteps - 1][1][2];
		for(int i = 0; i < numberOfTimeSteps - 1; i++) {
			for(int j = 0; j < 2; j++) {
				if(j==0) {
					derivativeLossFunctionEta[i][0][j] = derivativeLossFunctionEta1[i];
				}else {
					derivativeLossFunctionEta[i][0][j] = derivativeLossFunctionEta2[i];
				}
				
			}
		}
		double[][][] derivativeEtaOutput2 = new double[numberOfTimeSteps - 1][numberOfSecondHiddenLayerNeurons][2];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < numberOfSecondHiddenLayerNeurons; i++) {
				for(int j = 0; j < 2; j++) {
					if(j==0) {
						derivativeEtaOutput2[k][i][j] = weightMatrix3[k][i+1][j];
					}else {
					if(output3[k][1]==0) {
						derivativeEtaOutput2[k][i][j] = 0.0;
					}else {
					derivativeEtaOutput2[k][i][j] = weightMatrix3[k][i+1][j];
					}
					}
				}
			}
		}
		
		double[][][] derivativeOutput2Output1 = new double[numberOfTimeSteps - 1][numberOfFirstHiddenLayerNeurons][numberOfSecondHiddenLayerNeurons];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < numberOfFirstHiddenLayerNeurons; i++) {
				for(int j = 0; j < numberOfSecondHiddenLayerNeurons; j++) {
					if(output2[k][j+1]==0) {
						derivativeOutput2Output1[k][i][j] = 0.0;
					}else {
						derivativeOutput2Output1[k][i][j] = weightMatrix2[k][i+1][j];
					}
				}
			}
		}
		
		double[][][] derivativeOutput1WeightMatrix1 = new double[numberOfTimeSteps - 1][numberOfTimeSteps][numberOfFirstHiddenLayerNeurons];
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < k + 2; i++) {
				for(int j = 0; j < numberOfFirstHiddenLayerNeurons; j++) {
					if(getHiddenLayer1OutputBeforeReLU(inputVector, weightMatrix1)[k][j+1]>0) {
						derivativeOutput1WeightMatrix1[k][i][j] = inputVector[k][i];
					}else {
						derivativeOutput1WeightMatrix1[k][i][j] = -inputVector[k][i];
					}
				
				}
			}
		}
		
//		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
//					//derivativeMatrix[k][i][j] = derivativeOutput1WeightMatrix1[k][i][j]*multiplyMatrices(multiplyMatrices(derivativeLossFunctionEta, getTransformMatrix(derivativeEtaOutput2[k])),getTransformMatrix(derivativeOutput2Output1[k]))[0][j];
//					derivativeMatrix[k] = multiplyMatrices(derivativeOutput1WeightMatrix1[k],derivativeOutput2Output1[k]);
//		}
		double[][][] inputVector1 = new double [inputVector.length][1][inputVector[0].length];
		for(int j = 0; j < inputVector1.length; j++) {
			for(int k = 0; k < inputVector1[0].length; k++) {
				for(int l = 0; l < inputVector1[0][0].length; l++) {
					inputVector1[j][k][l] = inputVector[j][l];
				}
				
			}
		}
		
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			derivativeMatrix[k]= multiplyMatrices(getTransformMatrix(inputVector1[k]),multiplyMatrices(multiplyMatrices(derivativeLossFunctionEta[k],getTransformMatrix(derivativeEtaOutput2[k])),getTransformMatrix(derivativeOutput2Output1[k])));
		}
		for(int k = 0; k < (numberOfTimeSteps - 1); k++) {
			for(int i = 0; i < k + 2; i++) {
				for(int j = 0; j < numberOfFirstHiddenLayerNeurons; j++) {
					if(getHiddenLayer1OutputBeforeReLU(inputVector, weightMatrix1)[k][j+1]>0) {
						derivativeMatrix[k][i][j] = derivativeMatrix[k][i][j];
					}else {
						derivativeMatrix[k][i][j] = -derivativeMatrix[k][i][j];
					}
				
				}
			}
		}
		
		return derivativeMatrix;
	}	
}
