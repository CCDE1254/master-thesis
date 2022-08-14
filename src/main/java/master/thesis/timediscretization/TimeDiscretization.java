/**
 * 
 */
package master.thesis.timediscretization;

/**
 * @author QuanLiu
 *
 */

/*
 * this interface used to generate array of time discretization points
 */
public interface TimeDiscretization {

	public int getNumberOfTimePoints();
	
	public double getTimeStepSize();
	
	public double[] getTimeSeries();
	
}
