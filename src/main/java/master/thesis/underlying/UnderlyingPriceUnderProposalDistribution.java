/**
 * 
 */
package master.thesis.underlying;

import net.finmath.functions.NormalDistribution;

/**
 * @author QuanLiu
 *
 */
public class UnderlyingPriceUnderProposalDistribution {
	int numberOfSimulations;
	double initialPrice;
	double riskFreeRate;
	double volatility;
	double[] timeSeries;
	double[][] randomNumberMatrix;
	double[] naturalParametersOfProposalDistribution;




	public UnderlyingPriceUnderProposalDistribution(int numberOfSimulations, double initialPrice, double riskFreeRate,
			double volatility, double[] timeSeries, double[][] randomNumberMatrix,
			double[] naturalParametersOfProposalDistribution) {
		super();
		this.numberOfSimulations = numberOfSimulations;
		this.initialPrice = initialPrice;
		this.riskFreeRate = riskFreeRate;
		this.volatility = volatility;
		this.timeSeries = timeSeries;
		this.randomNumberMatrix = randomNumberMatrix;
		this.naturalParametersOfProposalDistribution = naturalParametersOfProposalDistribution;
	}


	public double[] transferNaturalParametersToNormalParameters() {
		double[] normalParameters = new double[2];
		double eta1 = naturalParametersOfProposalDistribution[0];
		double eta2 = naturalParametersOfProposalDistribution[1];
		normalParameters[0] = -2.0*eta1*eta2; //mean
		normalParameters[1] = -2.0*eta2; //variance
		return normalParameters;	
	}

	public double[][] getIncrementsMatrixUnderProposalDistribution(){
		double mean = transferNaturalParametersToNormalParameters()[0];
		double variance = transferNaturalParametersToNormalParameters()[1];
		double[][] incrementsMatrixUnderProposalDistribution = new double[timeSeries.length-1][numberOfSimulations];
		for(int j = 0; j < numberOfSimulations; j++) {
			for(int i = 0; i < timeSeries.length - 1 ; i++) {
				incrementsMatrixUnderProposalDistribution[i][j] = (NormalDistribution.inverseCumulativeDistribution(randomNumberMatrix[i][j])*variance + mean );
			}
		}
		return incrementsMatrixUnderProposalDistribution;
	}

	public double[][] getUnderlyingPriceMatrix() {
		double[][] incrementsMatrixUnderProposalDistribution = getIncrementsMatrixUnderProposalDistribution();
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
