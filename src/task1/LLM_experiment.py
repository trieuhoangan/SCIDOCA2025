import os
import sys
sys.path.append("/home/s2320037/SCIDOCA/SCIDOCA2025/src/utils")
import utils
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
query="We highlight our contributions by comparing with  . In  the state-wise ambiguity set is restricted to the following form:C s = {\u03bc s |\u03bc s (O i s ) \u2265 \u03b1 i s \u2200 i = 1, . . . , n s }, where \u03b1 i s \u2264 \u03b1 j s and O i s is a proper set of uncertain parameters with a \"nested-set\" structure, i.e., satisfying O i s \u2286 O j s , for all i < j [see Fig. 1(a) ]. This setup can effectively model distributions with a single mode (such as a Gaussian distribution), but less so when modeling multi-mode distributions such as a mixture Gaussian distribution. Moreover, other probabilistic information such as mean, variance etc. cannot be incorporated. Thus, in this technical note, we extend the distributionally robust MDP approach to handle ambiguity sets with more general structures. In particular, we consider a class of ambiguity sets, first proposed in  as a unifying framework for modeling and solving distributionally robust single-stage optimization problems, and embed them into the distributionally robust MDPs setup. These ambiguity sets are considerably more general: they are characterized by a class of O i s which can either be nested or disjoint [as shown in Fig. 1(b) ], and moreover, additional linear constraints are allowed to define the ambiguity set, which can be used to incorporate probabilistic information such as mean, covariance or other variation measures. We show that, under this more general class of ambiguity sets, the resulting distributionally robust MDPs remain tractable under mild technical conditions, and often outperform previous methods thanks to the fact that it can model uncertainty in a more flexible way."
candidates = {

            "7229756": {
            "title": "Distributionally robust Markov decision processes",
            "abstract": "We consider Markov decision processes where the values of the parameters are uncertain. This uncertainty is described by a sequence of nested sets (that is, each set contains the previous one), each of which corresponds to a probabilistic guarantee for a different confidence level so that a set of admissible probability distributions of the unknown parameters is specified. This formulation models the case where the decision maker is aware of and wants to exploit some (yet imprecise) a-priori information of the distribution of parameters, and arises naturally in practice where methods to estimate the confidence region of parameters abound. We propose a decision criterion based on distributional robustness: the optimal policy maximizes the expected total reward under the most adversarial probability distribution over realizations of the uncertain parameters that is admissible (i.e., it agrees with the a-priori information). We show that finding the optimal distributionally robust policy can be reduced to a standard robust MDP where the parameters belong to a single uncertainty set, hence it can be computed in polynomial time under mild technical conditions."
            },
            "16625241": {
            "title": "Distributionally robust convex optimization",
            "abstract": "Distributionally robust optimization is a paradigm for decision making under uncertainty where the uncertain problem data are governed by a probability distribution that is itself subject to uncertainty. The distribution is then assumed to belong to an ambiguity set comprising all distributions that are compatible with the decision maker's prior information. In this paper, we propose a unifying framework for modeling and solving distributionally robust optimization problems. We introduce standardized ambiguity sets that contain all distributions with prescribed conic representable confidence sets and with mean values residing on an affine manifold. These ambiguity sets are highly expressive and encompass many ambiguity sets from the recent literature as special cases. They also allow us to characterize distributional families in terms of several classical and/or robust statistical indicators that have not yet been studied in the context of robust optimization. We determine conditions under which distributionally robust optimization problems based on our standardized ambiguity sets are computationally tractable. We also provide tractable conservative approximations for problems that violate these conditions."
            },
            "6103434": {
            "title": "Robust Markov decision processes",
            "abstract": "Markov decision processes MDPs are powerful tools for decision making in uncertain dynamic environments. However, the solutions of MDPs are of limited practical use because of their sensitivity to distributional model parameters, which are typically unknown and have to be estimated by the decision maker. To counter the detrimental effects of estimation errors, we consider robust MDPs that offer probabilistic guarantees in view of the unknown parameters. To this end, we assume that an observation history of the MDP is available. Based on this history, we derive a confidence region that contains the unknown parameters with a prespecified probability 1-\u03b2. Afterward, we determine a policy that attains the highest worst-case performance over this confidence region. By construction, this policy achieves or exceeds its worst-case performance with a confidence of at least 1-\u03b2. Our method involves the solution of tractable conic programs of moderate size."
            },
            "486400": {
            "title": "Lightning does not strike twice: Robust MDPs with coupled uncertainty",
            "abstract": "We consider Markov decision processes under parameter uncertainty. Previous studies all restrict to the case that uncertainties among different states are uncoupled, which leads to conservative solutions. In contrast, we introduce an intuitive concept, termed \"Lightning Does not Strike Twice,\" to model coupled uncertain parameters. Specifically, we require that the system can deviate from its nominal parameters only a bounded number of times. We give probabilistic guarantees indicating that this model represents real life situations and devise tractable algorithms for computing optimal control policies."
            },
            "1537485": {
            "title": "Robust control of Markov decision processes with uncertain transition matrices",
            "abstract": "Optimal solutions to Markov decision problems may be very sensitive with respect to the state transition probabilities. In many practical problems, the estimation of these probabilities is far from accurate. Hence, estimation errors are limiting factors in applying Markov decision processes to real-world problems. ::: ::: We consider a robust control problem for a finite-state, finite-action Markov decision process, where uncertainty on the transition matrices is described in terms of possibly nonconvex sets. We show that perfect duality holds for this problem, and that as a consequence, it can be solved with a variant of the classical dynamic programming algorithm, the \"robust dynamic programming\" algorithm. We show that a particular choice of the uncertainty sets, involving likelihood regions or entropy bounds, leads to both a statistically accurate representation of uncertainty, and a complexity of the robust recursion that is almost the same as that of the classical recursion. Hence, robustness can be added at practically no extra computing cost. We derive similar results for other uncertainty sets, including one with a finite number of possible values for the transition matrices. ::: ::: We describe in a practical path planning example the benefits of using a robust strategy instead of the classical optimal strategy; even if the uncertainty level is only crudely guessed, the robust strategy yields a much better worst-case expected travel time."
            },
            "18980380": {
            "title": "Distributionally robust counterpart in Markov decision processes",
            "abstract": "This technical note studies Markov decision processes under parameter uncertainty. We adapt the distributionally robust optimization framework, assume that the uncertain parameters are random variables following an unknown distribution, and seek the strategy which maximizes the expected performance under the most adversarial distribution. In particular, we generalize a previous study [1] which concentrates on distribution sets with very special structure to a considerably more generic class of distribution sets, and show that the optimal strategy can be obtained efficiently under mild technical conditions. This significantly extends the applicability of distributionally robust MDPs by incorporating probabilistic information of uncertainty in a more flexible way."
            },
            "37925315": {
            "title": "Convex Optimization",
            "abstract": "Convex optimization problems arise frequently in many different fields. A comprehensive introduction to the subject, this book shows in detail how such problems can be solved numerically with great efficiency. The focus is on recognizing convex optimization problems and then finding the most appropriate technique for solving them. The text contains many worked examples and homework exercises and will appeal to students, researchers and practitioners in fields such as engineering, computer science, mathematics, statistics, finance, and economics."
            },
            "10308849": {
            "title": "Percentile optimization for Markov decision processes with parameter uncertainty",
            "abstract": "Markov decision processes are an effective tool in modeling decision making in uncertain dynamic environments. Because the parameters of these models typically are estimated from data or learned from experience, it is not surprising that the actual performance of a chosen strategy often differs significantly from the designer's initial expectations due to unavoidable modeling ambiguity. In this paper, we present a set of percentile criteria that are conceptually natural and representative of the trade-off between optimistic and pessimistic views of the question. We study the use of these criteria under different forms of uncertainty for both the rewards and the transitions. Some forms are shown to be efficiently solvable and others highly intractable. In each case, we outline solution concepts that take parametric uncertainty into account in the process of decision making."
            },
            "18576331": {
            "title": "A combined adaptive neural network and nonlinear model predictive control for multirate networked industrial process control",
            "abstract": "This paper investigates the multirate networked industrial process control problem in double-layer architecture. First, the output tracking problem for sampled-data nonlinear plant at device layer with sampling period $T_{d}$ is investigated using adaptive neural network (NN) control, and it is shown that the outputs of subsystems at device layer can track the decomposed setpoints. Then, the outputs and inputs of the device layer subsystems are sampled with sampling period $T_{u}$ at operation layer to form the index prediction, which is used to predict the overall performance index at lower frequency. Radial basis function NN is utilized as the prediction function due to its approximation ability. Then, considering the dynamics of the overall closed-loop system, nonlinear model predictive control method is proposed to guarantee the system stability and compensate the network-induced delays and packet dropouts. Finally, a continuous stirred tank reactor system is given in the simulation part to demonstrate the effectiveness of the proposed method."
            },
            "24341930": {
            "title": "Networked multirate output feedback control for setpoints compensation and its application to rougher flotation process",
            "abstract": "This paper investigates the setpoints compensation for a class of complex industrial processes. Plants at the device layer are controlled by the local regulation controllers, and a multirate output feedback control approach for setpoints compensation is proposed such that the local plants can reach the dynamically changed setpoints and the given economic objective can also be tracked via certain economic performance index (EPI). First, a sampled-data multivariable output feedback proportional integral (PI) controller is designed to regulate the performance of local plants. Second, the outputs and control inputs of the local plants at the device layer are sampled at operation layer sampling time to form the EPI. Thus, the multirate problem is solved by a lifting method. Third, the static setpoints are generated by real-time optimization and the dynamic setpoints are calculated by the compensator according to the error between the EPI and objective at each operation layer step. Then, a networked case is studied considering unreliable data transmission described by a stochastic packet dropout model. Finally, a rougher flotation process model is employed to demonstrate the effectiveness of the proposed method."
            }
        }

# content =     ""
# for id in candidates:
#     candidate_text = candidates[id]["title"]["abstract"]

candidate_text = json.dumps(candidates)

question = "Given a paragrapgh below and a list of papers, point out the id of papers that can be used as citations of the given paragraph\n\n Paragraph: {} \n\n List of Candidate papers: {} \n\n".format(query, candidate_text) 
messages = [
    {"role": "user", "content": question},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

outputs = model.generate(inputs, max_new_tokens=200)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(question)
