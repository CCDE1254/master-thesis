/**
 * 
 */
package master.thesis.timediscretization;

/**
 * @author QuanLiu
 *
 */

/*
 * this class used to generate time series with equal time step size
 */
public class TimeDiscretizationWithEqualTimeStepSize {

	double startTime;
	double endTime;
	int numberOfTimeSteps;
	
	
	public TimeDiscretizationWithEqualTimeStepSize(double startTime, double endTime, int numberOfTimeSteps) {
		super();
		this.startTime = startTime;
		this.endTime = endTime;
		this.numberOfTimeSteps = numberOfTimeSteps;
	}

	public int getNumberOfTimePoints() {
		return numberOfTimeSteps + 1;
	}

	public double getTimeStepSize() {
		double timeIntervalLength = endTime - startTime;
		double timeStepSize = timeIntervalLength / numberOfTimeSteps;
		return timeStepSize;
	}


	public double[] getTimeSeries() {
		double[] timeSeries = new double[getNumberOfTimePoints()];
		for(int i = 0; i < numberOfTimeSteps+1; i++) {
			timeSeries[i] = startTime + i * getTimeStepSize();
		}
		return timeSeries;
	}


}
