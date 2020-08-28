using System.Collections;
using System.Collections.Generic;
using UnityEditor.PackageManager.Requests;
using UnityEngine;
using System.Linq;
using UnityEditorInternal;

public class replay
{
    public List<double> states=new List<double>();
    public double reward;
    public replay(double ypos, double yvel, double r)
    {
        states.Add(ypos);
        states.Add(yvel);
        reward = r;
    }
}
public class brain1 : MonoBehaviour
{

    ANN ann;
    bool dead = false;
    float force = 10f;
    float reward;
    List<replay> replaymemory = new List<replay>();
    int mcapacity = 10000;
    float discount = 0.99f;
    float explorerate = 100.0f;
    float maxeplorerate = 100.0f;
    float minexplorerate = 0.01f;
    float exploredecay = 0.0001f;
    Vector3 startpos;
    // Start is called before the first frame update
    void Start()
    {
        ann = new ANN(2, 1, 1, 6, 0.3f);
        startpos = this.transform.position;
        Time.timeScale = 5.0f;
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown(KeyCode.Space))
        {
            Reset();
        }
    }
    private void FixedUpdate()
    {
        List<double> states = new List<double>();
        List<double> qs = new List<double>();
        states.Add(this.transform.position.y);
        states.Add(this.GetComponent<Rigidbody2D>().velocity.y);
        qs = ann.CalcOutput(states);
        this.GetComponent<Rigidbody2D>().AddForce(Vector2.up * force * (float)qs[0]);
        if (dead)
        {
            reward = -1;
        }
        else
        {
            reward = 0.1f;
        }
        replay lastmemory = new replay(this.transform.position.y, this.GetComponent<Rigidbody2D>().velocity.y, reward);
        if(replaymemory.Count>mcapacity)
        {
            replaymemory.RemoveAt(0);
        }
        replaymemory.Add(lastmemory);
        //Training And QLearning
        if(dead)
        {
            for (int i = replaymemory.Count - 1; i >= 0; i--)
            {
                List<double> toutputs_old = new List<double>();
                List<double> toutputs_new = new List<double>();
                toutputs_old = ann.CalcOutput(replaymemory[i].states);
                
                double feedback;
                if (i == replaymemory.Count - 1 || replaymemory[i].reward == -1)
                {
                    feedback = replaymemory[i].reward;
                }
                else
                {
                    toutputs_new = ann.CalcOutput(replaymemory[i + 1].states);
                    double maxQ = toutputs_new[0];
                    feedback = (replaymemory[i].reward + discount * maxQ);  //BELLMAN EQUATION
                }
                toutputs_old[0] = feedback;
                ann.Train(replaymemory[i].states, toutputs_old);
            }
            dead = false;
            Reset();
            replaymemory.Clear();
        }
    }
    private void OnCollisionEnter2D(Collision2D collision)
    {
        if(collision.gameObject.tag=="Finish")
        {
            dead = true;
        }
    }
    private void Reset()
    {
        this.transform.position = startpos;
        this.GetComponent<Rigidbody2D>().velocity = new Vector2(0, 0);
    }
}
