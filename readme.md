# ACM RecSys Challenge 2025 [ririka Team]
In RecSys Challenge '25, the method we used received 4th place in Academic Final Leaderboard and 10th place in Final Leaderboard.



# About the challenge
From the [Challenge](https://recsys.synerise.com/). <br><br>
The RecSys 2025 Challenge will be organized by Jacek Dąbrowski, Maria Janicka, Łukasz Sienkiewicz and Gergo Stomfai (Synerise), Dietmar Jannach (University of Klagenfurt, Austria), Marco Polignano ( University of Bari Aldo Moro, Italy), Claudio Pomo (Politecnico di Bari, Italy), Abhishek Srivastava (IIM Visakhapatnam, India), and Francesco Barile (Maastricht University, Netheralnds).

The challenge is designed to promote a unified approach to behavior modeling. Many modern enterprises rely on machine learning and predictive analytics for improved business decisions. Common predictive tasks in such organizations include recommendation, propensity prediction, churn prediction, user lifetime value prediction, and many others. A central piece of information that is used for these predictive tasks are logs of past behavior of users e.g., what they bought, what they added to their shopping cart, which pages they visited. Rather than treating these tasks as separate problems, we propose a unified modeling approach.

To achieve this, we introduce the concept of Universal Behavioral Profiles—user representations that encode essential aspects of each individual’s past interactions. These profiles are designed to be universally applicable across multiple predictive tasks, such as churn prediction and product recommendations. By developing representations that capture fundamental patterns in user behavior, we enable models to generalize effectively across different applications.


# Note: 
In order to improve the readability of the code, I changed some parts of the code (such as the names of variables and functions), so there is a slight difference between the code I used in the contest. This can lead to bugs in some places. If you find that the results are difficult to reproduce, or you find bugs that are difficult to fix, you can ask by email.

# Team members
We participated in the challenge as rirka, a team of 6 MSc from East China Normal University:

Minhao Wang <br>
Ruizhi Zhang <br>
Yichen Liu <br>
Yanyu Chen <br>
Yunhang He <br>
Jinghan Zhou <br>

We worked under the supervision of:

Wei Zhang (Professor) <br>
Wen Wu (Professor)



# Reproduce best submission


1. `conda create -n RecSys python==3.10.2`

2. `pip install -r requirements.txt`

3. `python -m data_utils.split_data --challenge-data-dir <your_challenge_data_dir>`

4. We made a series of feature engineering improvements based on the initial baseline. For details, please refer to the readme in the baseline.
Modify the path of run_baseline.sh and some paths in create_embeddings, then use:
`nohup run_baseline.sh & disown` to start it, and you will get the baseline embedding.

5. For the training part, the detailed process can be found in ./ubt_solution/README.
After modifying the data path, run the following commands separately:
`nohup bash run_category.sh & disown`
`nohup bash run_churn_category.sh & disown`
`nohup bash run_churn.sh & disown`
`nohup bash run_price.sh & disown`
to obtain the four embeddings.

6. Use the `sort` method in the `mege_embedding.ipynb` file to align the user embedding indices, then use the `concat` and `normalization` methods for `Feature fusion`.
