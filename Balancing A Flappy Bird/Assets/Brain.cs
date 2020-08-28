using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

//class to store past states for actions and associated rewards
public class Replay
{
    public List<double> states;
    public double reward;

	public Replay(double dtop, double dbot, double r)
	{
		states = new List<double>();
		states.Add(dtop);
		states.Add(dbot);
		reward = r;
	}
}

public class Brain : MonoBehaviour {
	
	public GameObject topBeam;
	public GameObject bottomBeam;
	ANN ann;

	float reward = 0.0f;							//reward to associate with actions
	List<Replay> replayMemory = new List<Replay>();	//memory - list of past actions and rewards
	int mCapacity = 10000;							//memory capacity
	
	float discount = 0.99f;							//how much future states affect rewards
	float exploreRate = 100.0f;						//chance of picking random action
	float maxExploreRate = 100.0f;					//max chance value
    float minExploreRate = 0.01f;					//min chance value
    float exploreDecay = 0.0001f;					//chance decay amount for each update

	int failCount = 0;								//count when the ball is dropped
	float moveForce = 0.5f;							//max angle to apply to tilting each update
													//make sure this is large enough so that the q value
													//multiplied by it is enough to recover balance
													//when the ball gets a good speed up
	float timer = 0;								//timer to keep track of balancing
	float maxBalanceTime = 0;						//record time ball is kept balanced	

	bool crashed = false;
	Vector3 startPos;
	Rigidbody2D rb;

	//states - INPUTS
	//platform x rotation
	//ball z position
	//ball velocity z

	//actions - OUTPUTS
	//rotate x direction positive
	//rotate x direction negative

	void Start () {
		ann = new ANN(2,2,1,6,0.2f);
		startPos = this.transform.position;
		Time.timeScale = 5.0f;
		rb = this.GetComponent<Rigidbody2D>();
	}

	/*GUIStyle guiStyle = new GUIStyle();
	void OnGUI()
	{
		guiStyle.fontSize = 25;
		guiStyle.normal.textColor = Color.white;
		GUI.BeginGroup (new Rect (10, 10, 600, 150));
		GUI.Box (new Rect (0,0,140,140), "Stats", guiStyle);
		GUI.Label(new Rect (10,25,500,30), "Fails: " + failCount, guiStyle);
		GUI.Label(new Rect (10,50,500,30), "Decay Rate: " + exploreRate, guiStyle);
		GUI.Label(new Rect (10,75,500,30), "Last Best Balance: " + maxBalanceTime, guiStyle);
		GUI.Label(new Rect (10,100,500,30), "This Balance: " + timer, guiStyle);
		GUI.EndGroup ();
	}*/

	void Update()
	{
		if(Input.GetKeyDown("space"))
			ResetBird();
	}

	void OnCollisionEnter2D(Collision2D col)
	{
		crashed = true;
	}

	void OnCollisionExit2D(Collision2D col)
	{
		crashed = false;
	}
	
	void FixedUpdate () {
		timer += Time.deltaTime;
		List<double> states = new List<double>();
		List<double> qs = new List<double>();
			
		states.Add(Vector3.Distance(this.transform.position,topBeam.transform.position));
		states.Add(Vector3.Distance(this.transform.position,bottomBeam.transform.position));
		
		qs = SoftMax(ann.CalcOutput(states));
		double maxQ = qs.Max();
		int maxQIndex = qs.ToList().IndexOf(maxQ);
		exploreRate = Mathf.Clamp(exploreRate - exploreDecay, minExploreRate, maxExploreRate);

		//if(Random.Range(0,100) < exploreRate)
		//	maxQIndex = Random.Range(0,2);

		if(maxQIndex == 0)
			rb.AddForce(Vector3.up * moveForce *(float)qs[maxQIndex]);
		else if (maxQIndex == 1)
			rb.AddForce(Vector3.up * -moveForce * (float)qs[maxQIndex]);
		
		if(crashed)
			reward = -1.0f;
		else
			reward = 0.1f;

		Replay lastMemory = new Replay(Vector3.Distance(this.transform.position,topBeam.transform.position),
								Vector3.Distance(this.transform.position,bottomBeam.transform.position),
								reward);

		if(replayMemory.Count > mCapacity)
			replayMemory.RemoveAt(0);
		
		replayMemory.Add(lastMemory);

		if(crashed) 
		{
			for(int i = replayMemory.Count - 1; i >= 0; i--)
			{
				List<double> toutputsOld = new List<double>();
				List<double> toutputsNew = new List<double>();
				toutputsOld = SoftMax(ann.CalcOutput(replayMemory[i].states));	

				double maxQOld = toutputsOld.Max();
				int action = toutputsOld.ToList().IndexOf(maxQOld);

			    double feedback;
				if(i == replayMemory.Count-1 || replayMemory[i].reward == -1)
					feedback = replayMemory[i].reward;
				else
				{
					toutputsNew = SoftMax(ann.CalcOutput(replayMemory[i+1].states));
					maxQ = toutputsNew.Max();
					feedback = (replayMemory[i].reward + 
						discount * maxQ);
				} 

				toutputsOld[action] = feedback;
				ann.Train(replayMemory[i].states,toutputsOld);
			}
		
			if(timer > maxBalanceTime)
			{
			 	maxBalanceTime = timer;
			} 

			timer = 0;

			crashed = false;
			ResetBird();
			replayMemory.Clear();
			failCount++;
		}	
	}

	void ResetBird()
	{
		this.transform.position = startPos;
		this.GetComponent<Rigidbody2D>().velocity = new Vector3(0,0,0);
	}

	List<double> SoftMax(List<double> oSums) 
    {
      double max = oSums.Max();

      float scale = 0.0f;
      for (int i = 0; i < oSums.Count; ++i)
        scale += Mathf.Exp((float)(oSums[i] - max));

      List<double> result = new List<double>();
      for (int i = 0; i < oSums.Count; ++i)
        result.Add(Mathf.Exp((float)(oSums[i] - max)) / scale);

      return result; 
    }
}
