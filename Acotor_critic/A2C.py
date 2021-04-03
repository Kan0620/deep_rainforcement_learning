#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 18:53:21 2021

@author: nakaharakan
"""

'''
class environment()で環境クラスを定義する 

.reset()で初期化してその状態を返す


.step(action=(行動のindex))で
next_s(その行動をすると得られる次のs,ニューラルネットに入れる状態にしておく)、
reward、done(エピソードが終わったかの真理値終わったらTrue)を返す 
=================================================================

class model() でニューラルネットを定義

.predict(入力のnumpy配列)で（N,n_output)の出力を返す
.save(保存名)でニューラルネット保存
.fit()で学習 入力は下で説明する.create_train_data()の返り値とcritic_loss_coef,entropy_coef
critic_loss_coefはTD誤差にかける係数、entropy_coefは方策のエントロピーにかける係数

pytorchで実装
==================================================================

class A2C_agent()でエージェントを定義する。引数はmodelインスタンス、environmentインスタンス
、actionの数n_action、割引率g、モデルを保存するときの名前save_nameと
下で説明するn_count,n_test,finish_score,game_over_r,n_step,critic_loss_coef,entropy_coef


.create_train_data()でn_count回エピソードが終わるまで繰り返す
状態sを表すarray　s
,実際に選ばれた行動の選ばれる確率のarray pi_a ,rのarrayを作り
n_stepにわたって実際の行動価値のarray qを作り（game_over付近は注意)
s,pi_a,qを返す



.test()で今のエージェントがn_test回のエピソードで一回あたり平均どれだけの利得を得られたかを返す

.save()でモデルを保存 モデル名は引数からのsave_name

.fit()でepochと.test()の値を表示しながら学習save()も毎epochする



'''

import numpy as np

class A2C_agent():
    
    def __init__(self,model,env,n_action,n_step,critic_loss_coef,entropy_coef,\
                 g,save_name,n_count,n_test,finish_score,game_over_r):
        
        self.model=model
        self.env=env
        self.n_action=n_action
        self.n_step=n_step
        self.critic_loss_coef=critic_loss_coef
        self.entropy_coef=entropy_coef
        self.g=g
        self.save_name=save_name
        self.n_count=n_count
        self.n_test=n_test
        self.finish_score=finish_score
        self.game_over_r=game_over_r
        
        
    def create_train_data(self):
        
        s=[]
        V_s=[]
        a_index=[]
        r=[]
        Q=[]
        
        for i in range(self.n_count):#n_count回探索
            
            done=False
            
            obs=self.env.reset()
            
            while not done:
                
                s.append(obs)#状態sを保存
                
                V,act=self.model.predict(obs.reshape((1,-1)))
                #状態価値と確率方策取得
                
                V=V[0][0]#V(s)
                
                V_s.append(V)
                
                act=act[0]
                
                action_index=np.random.choice(self.n_action,p=act)#行動を確率的に決定
                
                obs,reward,done=self.env.step(action_index)
                
                a_index.append(action_index)#行動のindexを保存
                
                r.append(reward)#即時報酬保存
                
                
        s=np.array(s)
        V_s=np.array(V_s)
        a_index=np.array(a_index)
        r=np.array(r)
        
        for i in range(len(s)-self.n_step):#n_step行動価値計算
            
            q=0
            
            r_s=r[i:i+self.n_step]
            
            if any([one_game_over_r in r_s for one_game_over_r in self.game_over_r]):
                #n_stepの間にgameoverしている場合
                
                game_over_flag=0
                count=0
                
                while game_over_flag==0:#game_overのrまで割引報酬を足す
                    
                    now_r=r_s[count]
                    
                    q+=np.power(self.g,count)*now_r
                    
                    if now_r in self.game_over_r:
                        
                        
                        game_over_flag=1
                        
                    count+=1
                
            else:
                
                for j,one_r in enumerate(r_s):#割引報酬和の計算
                    
                    q+=np.power(self.g,j)*one_r
                                        
                q+=np.power(self.g,self.n_step)*V_s[i+self.n_step]#割引した状態価値を足す
                
                
            Q.append(q)
        
        Q=np.array(Q)
        
                
        return s[:len(s)-self.n_step],a_index[:len(s)-self.n_step],Q
    
    def test(self):
        
        r=0
        
        for i in range(self.n_test):
            
            done=False
            
            obs=self.env.reset()
            
            
            while not done:
                
                V,act=self.model.predict(obs.reshape((1,-1)))
                
                act=act[0]
                
                action_index=np.random.choice(self.n_action,p=act)
                
                obs,reward,done=self.env.step(action_index)
                
                r+=reward
                
            
                
        return r/self.n_test
    
    def fit(self):
        
        epoch=1
        score=self.finish_score-1
        
        while score<self.finish_score:
            
            s,a_index,Q=self.create_train_data()
            
            
            self.model.fit(s,a_index,Q,self.critic_loss_coef,self.entropy_coef)
            
            score=self.test()
            
            self.model.save(self.save_name)
            
            print('epoch:'+str(epoch)+'  score:'+str(score))
            
            epoch+=1
            
            
            
            
            
    
            
        
            
        
                
                
                
                
                
                
                
                
    
        
        



















################ cart_poleでの使用例 ##############
        
import gym

class environment():
    
    def __init__(self):
        
        self.env=gym.make('CartPole-v0')
        
        self.env.reset()
        
        self.count=0
        
    def reset(self):
        
        self.count=0
        
        return self.env.reset()
    
    def step(self,action_index):
        
        observation,reward,done,info=self.env.step(action_index)
        
        self.count+=1
        
        if reward==1:
            
            reward=0
            
        if done:
            
            reward=-1
            
        if self.count==200:
            
            reward=1
            
            
        return observation,reward,done

import torch
from torch import tensor,LongTensor
from torch.nn import functional as F,Linear,Module,utils
from torch.optim import Adam


        
class model():
    
    def __init__(self):
        
        class Net(Module):
            
            def __init__(self):
                      
                super(Net,self).__init__()
                self.fc1=Linear(4,10)
                self.fc2=Linear(10,10)
                self.critic=Linear(10,1)
                self.actor=Linear(10,2)
                
                  
        
            def forward(self,x):
        
                h1=F.relu(self.fc1(x))
                h2=F.relu(self.fc2(h1))
                V=self.critic(h2)
                act=F.softmax(self.actor(h2),dim=1)
        
                return V,act
        
        
       
        self.net=Net()
        print(self.net)
        self.optim=Adam(self.net.parameters(),lr=0.01)
       
        
    def fit(self,s,a_index,Q,critic_loss_coef,entropy_coef):
        
        self.net.train()
        
        s=tensor(s,dtype=float)
        a_index=LongTensor(a_index.reshape((-1,1)))
        Q=tensor(Q,dtype=float)
        
        output_V,output_pi=self.net(s.float())#V,π取得
        
        log_prob=(output_pi.gather(1,a_index).log()).view(-1)#log方策計算
        
        adv=Q-output_V.view(-1)#アドバンテージ関数取得
        
        actor_loss=-(adv.detach()*log_prob).mean()#方策勾配定理よりactorのloss計算
         
        critic_loss=critic_loss_coef*adv.pow(2).mean()#二乗誤差からcriticのloss計算
        
        entropy=entropy_coef*(output_pi*output_pi.log()).sum(axis=1).mean()#方策のエントロピー計算
        
        total_loss=actor_loss+critic_loss-entropy
        
        self.optim.zero_grad()
        total_loss.backward()
        utils.clip_grad_norm(self.net.parameters(),0.5)#更新を抑える
        self.optim.step()
        
        
            
    def predict(self,x):
        
        self.net.eval()
        
        with torch.no_grad():
        
            x=tensor(x,dtype=float)
        
            V,act=self.net(x.float())
        
        return V.detach().numpy(),act.detach().numpy()
    
    
    def save(self,name=str()):
        
        torch.save(self.net.state_dict(),name)
        
        
        
        
        
        
        
def main():
    
    agent=A2C_agent(model=model(),env=environment(),n_action=2,n_step=5,g=0.9,save_name='A2C',\
    critic_loss_coef=0.5,entropy_coef=0.01,n_count=200,n_test=5,finish_score=1,game_over_r=[-1,1])
    
    agent.fit()
    
if __name__=='__main__':
    
    main()

        
        
        
        


    