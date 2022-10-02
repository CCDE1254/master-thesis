/**
 * 
 */
package master.thesis.gradient.descent;

import master.thesis.importance.sampling.DoubleBarrierKnockOutCallImportanceSampling;
import master.thesis.underlying.UnderlyingPrice;

/**
 * @author QuanLiu
 *
 */
public class AdaGradISCall {
	int numberOfSimulations;
	int numberOfTimeSteps;
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
	
	double learningRate;
	int numberOfIterationTimes;
	double epsilon;


	

	public AdaGradISCall(int numberOfSimulations, int numberOfTimeSteps,
			double[][] randomNumberMatrix, double initialStockPrice, double riskFreeRate, double volatilityTerm,
			double[] timeSeries, double strike, double maturity, double upperBoundFactorB,
			double upperBoundExponentialDelta1, double lowerBoundFactorA, double lowerBoundExponentialDelta2,
			double learningRate, int numberOfIterationTimes, double epsilon) {
		super();
		this.numberOfSimulations = numberOfSimulations;
		this.numberOfTimeSteps = numberOfTimeSteps;
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
		this.learningRate = learningRate;
		this.numberOfIterationTimes = numberOfIterationTimes;
		this.epsilon = epsilon;
	}

	public double[] getOptimalEta() {
		double eta1 = 0.0;
		double eta2 = -0.5;
		UnderlyingPrice normalDistributedRandomNumber = new UnderlyingPrice(numberOfSimulations, initialStockPrice, riskFreeRate, volatilityTerm,
				timeSeries, randomNumberMatrix);
		double[][] normalDistributedRandomNumberMatrix = normalDistributedRandomNumber.getNormalDistributedRandomNumberMatrixUnderTargetDistribution();
		double[][] underlyingPriceMatrix = normalDistributedRandomNumber.getUnderlyingPriceMatrix();
		double sumOfSquaredDerivativeEta1 = 0.0;
		double sumOfSquaredDerivativeEta2 = 0.0;
		for(int i = 0; i < numberOfIterationTimes; i++) {
			sumOfSquaredDerivativeEta1 += Math.pow(derivativeLossFunctionEta1(eta1,eta2,normalDistributedRandomNumberMatrix,underlyingPriceMatrix,i),2.0);
			eta1 -= (learningRate/Math.sqrt(epsilon+sumOfSquaredDerivativeEta1))*derivativeLossFunctionEta1(eta1,eta2,normalDistributedRandomNumberMatrix,underlyingPriceMatrix,i);
			sumOfSquaredDerivativeEta2 += Math.pow(derivativeLossFunctionEta2(eta1,eta2,normalDistributedRandomNumberMatrix,underlyingPriceMatrix,i),2.0);
			eta2 -= (learningRate/Math.sqrt(epsilon+sumOfSquaredDerivativeEta2))*derivativeLossFunctionEta2(eta1,eta2,normalDistributedRandomNumberMatrix,underlyingPriceMatrix,i);
	    }
		double[] eta = {eta1, eta2};
		return eta;
	}

	private double derivativeLossFunctionEta1(double eta1, double eta2,
			double[][] normalDistributedRandomNumberMatrix, double[][] underlyingPriceMatrix, int numberOfIteration) {
		double[] randomNumbers = new double[normalDistributedRandomNumberMatrix.length];
		for(int i = 0; i < normalDistributedRandomNumberMatrix.length; i++) {
			randomNumbers[i] = normalDistributedRandomNumberMatrix[i][numberOfIteration];
		}
		double payoffValid = 1.0;
		double sumOfRealizations = 0.0;
		double sumOfSquaredRealizations = 0.0;
		for(int i = 0; i < timeSeries.length-1; i++) {
			sumOfRealizations += randomNumbers[i];
			sumOfSquaredRealizations += randomNumbers[i]*randomNumbers[i];
		}
		for(int i = 0; i < timeSeries.length; i++) {
			double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
			double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
			if((underlyingPriceMatrix[i][numberOfIteration]>=upperBound) || (underlyingPriceMatrix[i][numberOfIteration]<=lowerBound)) {
				payoffValid = 0.0;
			}
		}
		return Math.exp(-2.0*riskFreeRate*maturity)*payoffValid*Math.max(0.0, underlyingPriceMatrix[timeSeries.length-1][numberOfIteration]-strike)
				*Math.max(0.0, underlyingPriceMatrix[timeSeries.length-1][numberOfIteration]-strike)
				*Math.pow(-2.0*eta2, 0.5*numberOfTimeSteps)
				*Math.exp(numberOfTimeSteps*eta1*eta1/(4.0*eta2) - eta1*sumOfRealizations - (0.5+eta2)*sumOfSquaredRealizations)
				*(numberOfTimeSteps*eta1/(2.0*eta2) - sumOfRealizations);
	}

	private double derivativeLossFunctionEta2(double eta1, double eta2,
			double[][] normalDistributedRandomNumberMatrix, double[][] underlyingPriceMatrix, int numberOfIteration) {
		double[] randomNumbers = new double[normalDistributedRandomNumberMatrix.length];
		for(int i = 0; i < normalDistributedRandomNumberMatrix.length; i++) {
			randomNumbers[i] = normalDistributedRandomNumberMatrix[i][numberOfIteration];
		}
		double payoffValid = 1.0;
		double sumOfRealizations = 0.0;
		double sumOfSquaredRealizations = 0.0;
		for(int i = 0; i < timeSeries.length-1; i++) {
			sumOfRealizations += randomNumbers[i];
			sumOfSquaredRealizations += randomNumbers[i]*randomNumbers[i];
		}
		for(int i = 0; i < timeSeries.length; i++) {
			double upperBound = upperBoundFactorB*Math.exp(upperBoundExponentialDelta1*timeSeries[i]);
			double lowerBound = lowerBoundFactorA*Math.exp(lowerBoundExponentialDelta2*timeSeries[i]);
			if((underlyingPriceMatrix[i][numberOfIteration]>=upperBound) || (underlyingPriceMatrix[i][numberOfIteration]<=lowerBound)) {
				payoffValid = 0.0;
			}
		}
		return -2.0*Math.exp(-2.0*riskFreeRate*maturity)*payoffValid*Math.max(0.0, underlyingPriceMatrix[timeSeries.length-1][numberOfIteration]-strike)
				*Math.max(0.0, underlyingPriceMatrix[timeSeries.length-1][numberOfIteration]-strike)
				*(0.5*numberOfTimeSteps)*Math.pow(-2.0*eta2, 0.5*numberOfTimeSteps-1.0)
				*Math.exp(numberOfTimeSteps*eta1*eta1/(4.0*eta2) - eta1*sumOfRealizations - (0.5+eta2)*sumOfSquaredRealizations)
			   +Math.exp(-2.0*riskFreeRate*maturity)*payoffValid*Math.max(0.0, underlyingPriceMatrix[timeSeries.length-1][numberOfIteration]-strike)
				*Math.max(0.0, underlyingPriceMatrix[timeSeries.length-1][numberOfIteration]-strike)
				*Math.pow(-2.0*eta2, 0.5*numberOfTimeSteps)
				*Math.exp(numberOfTimeSteps*eta1*eta1/(4.0*eta2) - eta1*sumOfRealizations - (0.5+eta2)*sumOfSquaredRealizations)
				*(-numberOfTimeSteps*eta1*eta1/(4.0*eta2*eta2) - sumOfSquaredRealizations);
	}


}
