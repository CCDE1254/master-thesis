/**
 * 
 */
package master.thesis.underlying;

import net.finmath.functions.NormalDistribution;

/**
 * @author QuanLiu
 *
 */
public class UnderlyingPriceUnderProposalDistributions {
	int numberOfSimulations;
	double initialPrice;
	double riskFreeRate;
	double volatility;
	double[] timeSeries;
	double[][] randomNumberMatrix;
	double[][] naturalParametersOfProposalDistribution;




	public UnderlyingPriceUnderProposalDistributions(int numberOfSimulations, double initialPrice, double riskFreeRate,
			double volatility, double[] timeSeries, double[][] randomNumberMatrix,
			double[][] naturalParametersOfProposalDistribution) {
		super();
		this.numberOfSimulations = numberOfSimulations;
		this.initialPrice = initialPrice;
		this.riskFreeRate = riskFreeRate;
		this.volatility = volatility;
		this.timeSeries = timeSeries;
		this.randomNumberMatrix = randomNumberMatrix;
		this.naturalParametersOfProposalDistribution = naturalParametersOfProposalDistribution;
	}


	public double[][] transferNaturalParametersToNormalParameters() {
		double[][] normalParameters = new double[timeSeries.length-1][2];
		for(int i = 0; i < timeSeries.length - 1 ; i++) {
			double eta1 = naturalParametersOfProposalDistribution[i][0];
			double eta2 = naturalParametersOfProposalDistribution[i][1];
			normalParameters[i][0] = -2.0*eta1*eta2; //mean
			normalParameters[i][1] = -2.0*eta2; //variance
		}	
		return normalParameters;	
	}

	public double[][] getIncrementsMatrixUnderProposalDistributions(){
		double[][] incrementsMatrixUnderProposalDistribution = new double[timeSeries.length-1][numberOfSimulations];
		for(int j = 0; j < numberOfSimulations; j++) {
			for(int i = 0; i < timeSeries.length - 1 ; i++) {
				double mean = transferNaturalParametersToNormalParameters()[i][0];
				double variance = transferNaturalParametersToNormalParameters()[i][1];
				incrementsMatrixUnderProposalDistribution[i][j] = (NormalDistribution.inverseCumulativeDistribution(randomNumberMatrix[i][j])*variance + mean );
			}
		}
		return incrementsMatrixUnderProposalDistribution;
	}

	public double[][] getUnderlyingPriceMatrix() {
		double[][] incrementsMatrixUnderProposalDistribution = getIncrementsMatrixUnderProposalDistributions();
		double[][] underlyingPriceMatrix = new double[timeSeries.length][numberOfSimulations];
		for(int i = 0; i < numberOfSimulations; i++) {
			underlyingPriceMatrix[0][i] = initialPrice;
		}
		for(int j = 0; j < numberOfSimulations; j++) {
			for(int i = 1; i < timeSeries.length; i++) {
				underlyingPriceMatrix[i][j] = underlyingPriceMatrix[i-1][j] * Math.exp( (riskFreeRate - (1.0/2)*Math.pow(volatility, 2))*(timeSeries[i]-timeSeries[i-1]) + 
						volatility*Math.sqrt(timeSeries[i]-timeSeries[i-1])*(incrementsMatrixUnderProposalDistribution[i-1][j]));
			}
		}
		return underlyingPriceMatrix;
	}
}
