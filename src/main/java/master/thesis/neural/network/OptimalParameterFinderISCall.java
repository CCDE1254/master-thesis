/**
 * 
 */
package master.thesis.neural.network;

import master.thesis.importance.sampling.DoubleBarrierKnockOutCallImportanceSampling;

/**
 * @author QuanLiu
 *
 */
public class OptimalParameterFinderISCall {
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
	
	int numberOfPossibleValues;
	double minEta1;
	double maxEta1;
	double minEta2;
	double maxEta2;
	

	
	public OptimalParameterFinderISCall(int numberOfSimulations, int numberOfTimeSteps, double[][] randomNumberMatrix, double initialStockPrice,
			double riskFreeRate, double volatilityTerm, double[] timeSeries, double strike, double maturity,
			double upperBoundFactorB, double upperBoundExponentialDelta1, double lowerBoundFactorA,
			double lowerBoundExponentialDelta2, int numberOfPossibleValues, double minEta1, double maxEta1,
			double minEta2, double maxEta2) {
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
		this.numberOfPossibleValues = numberOfPossibleValues;
		this.minEta1 = minEta1;
		this.maxEta1 = maxEta1;
		this.minEta2 = minEta2;
		this.maxEta2 = maxEta2;
	}



	public double[] getOptimalEta() {
		double[] listOfStandardError = new double[numberOfPossibleValues*numberOfPossibleValues];
		int i = 0;
		double[][] etaList = new double[numberOfPossibleValues*numberOfPossibleValues][2];
		for(double eta1 = minEta1; eta1 <= maxEta1; eta1+=(maxEta1-minEta1)/(numberOfPossibleValues-1)) {
			for(double eta2 = minEta2; eta2 <= maxEta2; eta2+=(maxEta2-minEta2)/(numberOfPossibleValues-1)) {
				double[] eta = {eta1,eta2};
				DoubleBarrierKnockOutCallImportanceSampling callIS = new DoubleBarrierKnockOutCallImportanceSampling(numberOfSimulations, numberOfTimeSteps,
						randomNumberMatrix,
						initialStockPrice, riskFreeRate, volatilityTerm, timeSeries, strike,
						maturity, upperBoundFactorB, upperBoundExponentialDelta1, lowerBoundFactorA,
						lowerBoundExponentialDelta2, eta);
				listOfStandardError[i] = callIS.getStandardError();
				etaList[i][0] = eta1;
				etaList[i][1] = eta2;
				i++;
			}
		}
		int smallestIndex = indexOfSmallestElement(listOfStandardError);
		return etaList[smallestIndex];
	}
	
	public static int indexOfSmallestElement(double[] numbers){
        double min=10000;
        int minIndex=0;
        for (int i=0;i<numbers.length;i++){
            if (min>numbers[i]){
                min=numbers[i];
                minIndex=i;
            }
        }
        return minIndex;
    }

	
}
