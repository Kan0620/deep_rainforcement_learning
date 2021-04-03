#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 18:56:30 2021

@author: nakaharakan
"""
import numpy as np
import random


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
.fit(入力のnumpy配列,出力のnumpy配列)で学習

pytorchで実装

==================================================================

class duelingDQN_agent()でエージェントを定義する。引数はmodelインスタンス、environmentインスタンス
、actionの数n_action、学習率alpha、割引率g、モデルを保存するときの名前save_nameと
下で説明するn_count,n_test,finish_score,using_data_rate,game_over_r

.new_epsilon(epoch)でそのepochでのepsilonを返す

.create_train_data(epsilon,count)でn_count回エピソードが終わるまでepsilon-greedyで探索
sのarray,aのindexのarray ,rのarray,s'のarrayをを作りDQNのアルゴリズムにしたがって
教師信号の入力xと出力yを返す。この時データの時系列を破壊するため得られた履歴のusing_data_rate*100%だけ使用

self.next_Q_predict（次のs’を表すNNへの入力,そのs’に到達した時に得られたr)で
shapeが（入力されたs'のlen,行動数)のQ値を返すがrがgame_over_rの場合はその行のQ値
を全て0にする ゲームオーバーがないタスクの場合はとり得ないrを指定しておけばOK

.test()で今のエージェントが100%自分で行動選択した場合n_test回のエピソードで
一回あたり平均どれだけの利得を得られたかを返す

.save()でモデルを保存 モデル名は引数からのsave_name

.fit()でepochと.test()の値を表示しながら学習、test()の値がfinish_score以上になれば
学習終了、saveも毎epochする

simpleDQNとの違いがネットワーク部分だけなので使用例部分しか変わってません

'''

class duelingDQN_agent():
    
    def __init__(self,model,env,n_action,alpha,g,n_count,using_data_rate,\
                 game_over_r,n_test,finish_score,save_name):
        
        
        print(model)
        
        self.model=model
        
        self.env=env
        
        self.n_action=n_action
        
        self.alpha=alpha
        
        self.g=g
        
        self.n_count=n_count
        
        self.using_data_rate=using_data_rate
        
        self.n_test=n_test
        
        self.finish_score=finish_score
        
        self.save_name=save_name
        
        self.game_over_r=game_over_r
        
    def new_epsilon(self,epoch): 
        
        #指数的に減衰させる
        
        return 0.95**(epoch-1)
    
    def next_Q_predict(self,next_s,r):
        
        y=self.model.predict(next_s)
        
        game_over_index=np.where(sum([r==i for i in self.game_over_r]))[0]
        
        y[game_over_index]=np.zeros(self.n_action)
        
        return y
    
    def create_train_data(self,epsilon):
        
        s=[]
        a_index=[]
        r=[]
        next_s=[]
        
        #n_count回分のデータ作成
        
        for i in range(self.n_count):
            
            #エピソードが終わっているかの変数doneをFalseに初期化、環境も初期化
            
            done=False
            
            observation=self.env.reset()
            
            while not done:                
                
                if random.random()<self.using_data_rate:#今からの処理で得られるデータが学習に使われる場合
                    
                
                    
                    s.append(observation)
                
                    if random.random()<epsilon:#探索
                        
                            action=np.random.choice(self.n_action)
                        
                    else:#活用
                         
                            action=np.argmax(self.model.predict(np.array([observation]))[0])
                        
                    a_index.append(action)
                        
                    observation,reward,done=self.env.step(action)
                
                    r.append(reward)
                
                    next_s.append(observation)
                
        
                else:#使われない場合
                    if random.random()<epsilon:#探索
                        
                            action=np.random.choice(self.n_action)
                        
                    else:#活用
                         
                            action=np.argmax(self.model.predict(np.array([observation]))[0])
                    
                    observation,reward,done=self.env.step(action)
                    
        
        s=np.array(s)
        a_index=np.array(a_index)
        r=np.array(r)
        next_s=np.array(next_s)
        
        #x、yはニューラルネットの入力と出力
        x=s
        
        y=self.model.predict(s)
        
        #Q-learningの更新式でyを更新
        
        y[np.arange(len(x)),a_index]=\
        (1-self.alpha)*y[np.arange(len(x)),a_index]+\
        self.alpha*(r+self.g*self.next_Q_predict(next_s,r).max(axis=1))
                
                
        return x,y        
    
    
                
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
            x,y=self.create_train_data(epsilon)
            
            #ニューラルネット学習
            self.model.fit(x,y)
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
                self.to_adv=Linear(10,2)
                self.to_V=Linear(10,1)
        
            def forward(self,x):
        
                h1=F.relu(self.fc1(x))
                h2=F.relu(self.fc2(h1))
                adv=self.to_adv(h2)
                V=self.to_V(h2).expand(-1,2) 
                '''
                Vはshapeが（N,1)なのであとでadvと足し算できるように.expand(-1,行動の数=2)
                しています。
                '''
                
                outputs=V+adv-adv.mean(1,keepdim=True).expand(-1,2).detach()
                
                '''
                adv.mean(1）はshapeが（N)なのでkeepdim=Trueで（N,1)にして
                V+advと足し算できるように.expand(-1,行動の数=2)
                しています。
                '''
                
                
                
                return outputs
        
        
       
        self.net=Net()
        print(self.net)
        self.optim=Adam(self.net.parameters(),lr=0.01)
       
        
    def fit(self,x,y):
        
        x=tensor(x,dtype=float,requires_grad=True)
        y=tensor(y,dtype=float)
        fit_set=TensorDataset(x,y)
        fit_loader=DataLoader(fit_set,batch_size=32,shuffle=True)
        
        self.net.train()
        
        for data,targets in fit_loader:
            
            self.optim.zero_grad()
            
            outputs=self.net(data.float())
            
            loss=F.smooth_l1_loss(outputs,targets.float())
            
            loss.backward()
            self.optim.step()
            
    def predict(self,x):
        
        self.net.eval()
        
        x=tensor(x,dtype=float)
        
        
        return self.net(x.float()).detach().numpy()
    
    def save(self,name=str()):
        
        torch.save(self.net.state_dict(),name)
    
def main():
    
    
    
    agent=duelingDQN_agent(model=model(),
                    env=environment(),
                    n_action=2,
                    alpha=0.5,
                    g=0.9,
                    n_count=25,
                    using_data_rate=0.7,
                    game_over_r=[-1,1],
                    n_test=5,
                    finish_score=1,
                    save_name='duelingDQN')
    
    agent.fit()

            
if __name__=='__main__':
    
    main()

          