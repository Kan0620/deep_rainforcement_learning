#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 15:28:02 2021

@author: nakaharakan
"""


import numpy as np
import random



class prioritized_replay_agent():
    
    def __init__(self,model,env,n_action,n_train_data,td_error_epsilon,capacity,alpha,beta,lr_alpha,g,n_count,\
                 game_over_r,n_test,finish_score,save_name,max_beta_epoch):
        
        print(model)
        
        if n_train_data>capacity:
            
            print('n_train_data<capacityとなるようにしてください')
        
        self.model=model
        
        self.env=env
        
        self.n_action=n_action
        
        self.n_train_data=n_train_data
        
        self.td_error_epsilon=td_error_epsilon
        
        self.capacity=capacity
        
        self.alpha=alpha
        
        self.beta=beta
        
        self.lr_alpha=lr_alpha
        
        self.g=g
        
        self.n_count=n_count
        
        self.n_test=n_test
        
        self.finish_score=finish_score
        
        self.save_name=save_name
        
        self.game_over_r=game_over_r
        
        self.max_beta_epoch=max_beta_epoch
        
        #memoryの定義　一連の（s,α,r,s')でindexが揃っていることに注意
        
        obs=self.env.reset()
        
        self.memory_s=obs.reshape(1,-1)
        
        action=np.random.choice(self.n_action)
        
        obs,reward,done=self.env.step(action)
        
        self.memory_a_index=np.array(action)
        
        self.memory_r=np.array(reward)
        
        self.memory_next_s=obs.reshape(1,-1)
        
        self.TD_error=np.array([])
        
    def new_epsilon(self,epoch): 
        
        #指数的に減衰させる
        
        return 0.95**(epoch-1)
    
    def new_beta(self,epoch):
        
        if epoch<self.max_beta_epoch:
            
            return self.beta+(1-self.beta)*(epoch-1)/(self.max_beta_epoch-1)
        
        else:
            
            return 1
       
    
    
    def next_Q_predict(self,next_s,r):
        
        y=self.model.predict(next_s)
        
        game_over_index=np.where(sum([r==i for i in self.game_over_r]))[0]
        
        y[game_over_index]=np.zeros(self.n_action)
        
        return y
    
    
    def create_train_data(self,epsilon):
        
        
        #n_count回分のデータ作成
        
        for i in range(self.n_count):
            
            #エピソードが終わっているかの変数doneをFalseに初期化、環境も初期化
            
            done=False
            
            observation=self.env.reset()
            
            while not done:                
                
                
                    
                self.memory_s=np.append(self.memory_s,observation.reshape(1,-1),0)
                
                if random.random()<epsilon:#探索
                        
                    action=np.random.choice(self.n_action)
                        
                else:#活用
                         
                    action=np.argmax(self.model.predict(np.array([observation]))[0])
                        
                self.memory_a_index=np.append(self.memory_a_index,np.array(action))
                        
                observation,reward,done=self.env.step(action)
                
                self.memory_r=np.append(self.memory_r,np.array(reward))
                
                self.memory_next_s=np.append(self.memory_next_s,observation.reshape(1,-1),0)
                
                self.memory_done=np.append(self.memory_next_s,observation.reshape(1,-1),0)
                
        
        #x、yはニューラルネットの入力と出力
        new_Q=self.model.predict(self.memory_s)
        #Q-learningの更新式でyを更新
        
        new_Q[np.arange(len(self.memory_s)),self.memory_a_index]=\
        (1-self.lr_alpha)*new_Q[np.arange(len(self.memory_s)),self.memory_a_index]+\
        self.lr_alpha*(self.memory_r+self.g*self.next_Q_predict(self.memory_next_s,self.memory_r).max(axis=1))
        
        x=self.memory_s
        
        y=new_Q
        
        w=1
        
        
        if len(self.memory_s)>self.n_train_data:
            
            TD_error=np.power(np.abs(new_Q-self.model.predict(self.memory_s))[np.arange(len(self.memory_s)),self.memory_a_index]\
            +self.td_error_epsilon,self.alpha)
            
            train_index=np.random.choice(len(self.memory_s),self.n_train_data,\
                                         p=TD_error/TD_error.sum(),replace=False)
            
            w=(1/(len(TD_error)*TD_error/TD_error)).max()
            
            
            
            self.TD_error=TD_error
            
            x=x[train_index]
            y=y[train_index]
            
            
        return x,y,w
    
    
    
    def forget_memory(self):
        
        #TD誤差が小さいデータを優先的に忘れる
        
        if len(self.memory_s)>self.capacity:
            
            
            inverse_TD_error=1/(self.TD_error+1e-8)
            
            forget_index=np.random.choice(len(self.memory_s),len(self.memory_s)-self.capacity,\
                                         p=inverse_TD_error/inverse_TD_error.sum(),replace=False)
            
            self.memory_s=np.delete(self.memory_s,forget_index,axis=0)
            self.memory_a_index=np.delete(self.memory_a_index,forget_index)
            self.memory_r=np.delete(self.memory_r,forget_index)
            self.memory_next_s=np.delete(self.memory_next_s,forget_index,axis=0)
            
            
    def test(self):
        
        r=0
        
        for i in range(self.n_test):
            
            done=False
            
            observation=self.env.reset()
            
            while not done:
                
                                
                action=np.argmax(self.model.predict(np.array([observation]))[0])
                
                observation,reward,done=self.env.step(action)
                
                r+=reward
                
        return r/self.n_test
    
    
    def save(self,save_name):
        
        self.model.save(save_name)
        
        
    
    def fit(self):
        
        epoch=1
        
        score=self.finish_score-1
        
        while  self.finish_score>score: #学習の継続条件
            
            epsilon=self.new_epsilon(epoch)#新しいepsilon取得
            
            #教師データ取得
            x,y,w=self.create_train_data(epsilon)
            
            beta=self.new_beta(epoch)
            
            w=np.power(w,beta)
            
            print(w,beta)
            
            #ニューラルネット学習
            self.model.fit(x,y,w)
            
            #データ量がcapacityを超えてる場合は忘れる
            self.forget_memory()
            #モデル保存
            self.save(self.save_name)
            
            #現在のモデルの平均スコア取得
            score=self.test()
            
            print('epoch:'+str(epoch)+'  score:'+str(score))
            
            epoch+=1
            
        print('fitting finished')
    
    
       
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
from torch import tensor
from torch.utils.data import TensorDataset,DataLoader
from torch.nn import functional as F,Linear,Module
from torch.optim import Adam

        
class model():
    
    def __init__(self):
        
        class Net(Module):
            
            def __init__(self):
                      
                super(Net,self).__init__()
                self.fc1=Linear(4,10)
                self.fc2=Linear(10,10)
                self.fc3=Linear(10,2)
                  
        
            def forward(self,x):
        
                h1=F.relu(self.fc1(x))
                h2=F.relu(self.fc2(h1))
                outputs=self.fc3(h2)
        
                return outputs
        
        
       
        self.net=Net()
        print(self.net)
        self.optim=Adam(self.net.parameters(),lr=0.01)
       
        
        
    def fit(self,x,y,w):
        
        x=tensor(x,dtype=float)
        y=tensor(y,dtype=float)
        w=tensor(w,dtype=float)
        fit_set=TensorDataset(x,y)
        fit_loader=DataLoader(fit_set,batch_size=32,shuffle=True)
        
        self.net.train()
        
        for data,targets in fit_loader:
            
            self.optim.zero_grad()
            
            outputs=self.net(data.float())
            
            loss=F.smooth_l1_loss(outputs,targets.float())*w
            
            loss.backward()
            self.optim.step()
            
            
    def predict(self,x):
        
        self.net.eval()
        
        x=tensor(x,dtype=float)
        
        
        return self.net(x.float()).detach().numpy()
    
    def save(self,name=str()):
        
        torch.save(self.net.state_dict(),name)

    
def main():
    
    
    agent=prioritized_replay_agent(model=model(),
                                   env=environment(),
                                   n_action=2,
                                   n_train_data=1000,
                                   td_error_epsilon=0.0001,
                                   capacity=10000,
                                   alpha=0.5,
                                   beta=0.4,
                                   max_beta_epoch=100,
                                   lr_alpha=0.5,
                                   g=0.9,
                                   n_count=25,
                                   game_over_r=[1,-1],
                                   n_test=5,
                                   finish_score=1,
                                   save_name='prioritized_replay')
    
    agent.fit()

            
if __name__=='__main__':
    
    main()

                
    
    
    













