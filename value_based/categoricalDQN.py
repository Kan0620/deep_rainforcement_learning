#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 23:04:22 2021

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

class DQN_agent()でエージェントを定義する。引数はmodelインスタンス、environmentインスタンス、actionの数n_action、学習率alpha、割引率g、モデルを保存するときの名前save_nameと
下で説明するn_count,n_test,finish_score,using_data_rate,game_over_r,n_atoms,min_V,max_V

.new_epsilon(epoch)でそのepochでのepsilonを返す

.projected_Q（次のs’を表すNNへの入力,そのs’に到達した時に得られたr)で
r+γ＊maxQ(s',α')のQ値の分布を返す　ただしrはgame_over_rでないものに限る
ゲームオーバーがないタスクの場合はとり得ないrを指定しておけばOK

.game_over_projected_Q（r)でgame_over_rだった場合のQ(s,a)の分布を返す

.new_Q_distribution(next_sのarray,対応するrのarray)で
r+γ*Q(s',a')の分布のarrayをrで場合分けしながら上の.projected_Qと.game_over_projected_Q
を使って出力する


.create_train_data(epsilon,count)でn_count回エピソードが終わるまでepsilon-greedyで探索
sのarray,aのindexのarray ,rのarray,s'のarrayをを作りDQNのアルゴリズムにしたがって
教師信号の入力xと出力yを返す。この時データの時系列を破壊するため得られた履歴のusing_data_rate*100%だけ使用






.test()で今のエージェントが100%自分で行動選択した場合n_test回のエピソードで
一回あたり平均どれだけの利得を得られたかを返す

.save()でモデルを保存 モデル名は引数からのsave_name

.fit()でepochと.test()の値を表示しながら学習、test()の値がfinish_score以上になれば
学習終了、saveも毎epochする



'''




class categoricalDQN_agent():
    
    def __init__(self,model,env,n_action,alpha,g,n_atoms,min_V,max_V,n_count,using_data_rate,\
                 game_over_r,n_test,finish_score,save_name):
        
        print(model)
        
        self.model=model
        self.env=env
        self.n_action=n_action
        self.alpha=alpha
        self.g=g
        self.n_atoms=n_atoms
        self.min_V=min_V
        self.max_V=max_V
        self.q_array=np.linspace(min_V,max_V,n_atoms)
        self.delta_q=(max_V-min_V)/(n_atoms-1)
        self.n_count=n_count
        self.using_data_rate=using_data_rate
        self.game_over_r=game_over_r
        self.n_test=n_test
        self.finish_score=finish_score
        self.save_name=save_name
        
    def new_epsilon(self,epoch): 
        
        #指数的に減衰させる
        
        return 0.95**(epoch-1)
    
    def projected_Q(self,r,next_Q_dist):
        
        new_q_array=r+self.g*self.q_array
        
        projected_Q_dist=np.zeros(self.n_atoms)
        
        compare_index=(self.q_array[:,np.newaxis]-new_q_array[np.newaxis,:]<0)\
        .sum(axis=0)-1
        '''
        新しいビンのある要素についてそのq値が元のビンのq値のなかで0以下かつ最大となる元のビンのq値のindex
        例えば self.q_array＝array([-2, -1,  0,  1,  2])
              new_q_array=array([-0.8, -0.1,  1. ,  2. ,  2.8])なら
                
                -1<-0.8<0 でindexは1
                -1<-0.1<0 でindexは1
                0<1<=1で　indexは２
                1<2<=2でindexは３
                2<2.8でindexは4
                となって出力はarray([1, 1, 2, 3, 4])
                両隣の数のindexがわかるのであとは確率を割り振る
                
        '''
        
        
        for i,(index,new_q) in enumerate(zip(compare_index,new_q_array)):
            
            
            if index==-1:
                
                projected_Q_dist[0]+=next_Q_dist[i]
                
            elif index==self.n_atoms-1:
                
                projected_Q_dist[-1]+=next_Q_dist[i]
                
            else:
                
                delta_down=new_q-self.q_array[index]
                delta_up=self.q_array[index+1]-new_q
                
                projected_Q_dist[index]+=next_Q_dist[i]*delta_up/self.delta_q
                projected_Q_dist[index+1]+=next_Q_dist[i]*delta_down/self.delta_q
                
               
        return projected_Q_dist
    
    def game_over_projected_Q(self,r):
        
        
        compare_index=(self.q_array-r<0).sum()-1
        
        game_over_projected_Q_dist=np.zeros(self.n_atoms)
        
        if compare_index==-1:
                
            game_over_projected_Q_dist[0]+=1
                
        elif compare_index==self.n_atoms-1:
                
            game_over_projected_Q_dist[-1]+=1
                
        else:
                
            delta_down=r-self.q_array[compare_index]
            delta_up=self.q_array[compare_index+1]-r
                
            game_over_projected_Q_dist[compare_index]+=delta_up/self.delta_q
            game_over_projected_Q_dist[compare_index+1]+=delta_down/self.delta_q
            
        
            
        return game_over_projected_Q_dist
    
    def new_Q_distribution(self,next_s,r):
        
        y=self.model.predict(next_s)
        
        
        max_Q_index=np.argmax(np.dot(y,self.q_array),axis=1)
            
        
        next_Q_dist=y[np.arange(len(y)),max_Q_index]
        
        
        new_Q_dist=[]
        
        for one_next_Q_dist,one_r in zip(next_Q_dist,r):
            
            if one_r in self.game_over_r:
                
                
            
                new_Q_dist.append(self.game_over_projected_Q(one_r))
                
            else:
                
                new_Q_dist.append(self.projected_Q(one_r,one_next_Q_dist))
                
        new_Q_dist=np.array(new_Q_dist)
        
        
            
        return new_Q_dist
        
        
        
    
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
                        
                         #NNの出力（shape(n_action,n_atoms)）の行列とself.q_array（shape(n_atoms)）
                         #の行列積(shape(n_action))は行動価値Q(s,a)だからそのargmaxをとる
                            action=np.argmax(np.dot(self.model.predict(np.array([observation]))[0],self.q_array))
                        
                    a_index.append(action)
                        
                    observation,reward,done=self.env.step(action)
                
                    r.append(reward)
                
                    next_s.append(observation)
                
        
                else:#使われない場合
                    if random.random()<epsilon:#探索
                        
                            action=np.random.choice(self.n_action)
                        
                    else:#活用
                         
                            action=np.argmax(np.dot(self.model.predict(np.array([observation]))[0],self.q_array))
                    
                    observation,reward,done=self.env.step(action)
                    
        
        s=np.array(s)
        a_index=np.array(a_index)
        r=np.array(r)
        next_s=np.array(next_s)
        
        #x、yはニューラルネットの入力と出力
        x=s
        y=self.model.predict(s)
        
        #Q-learningの更新式でyを更新
        
        y[np.arange(len(x)),a_index]=self.alpha*y[np.arange(len(x)),a_index]+\
        (1-self.alpha)*self.new_Q_distribution(next_s,r)
                
        
        return x,y
        
    def test(self):
        
        r=0
        
        for i in range(self.n_test):
            
            done=False
            
            observation=self.env.reset()
            
            while not done:
                
                                
                action=action=np.argmax(np.dot(self.model.predict(np.array([observation]))[0],self.q_array))
                
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
                self.fc1=Linear(4,16)
                self.fc2=Linear(16,32)
                self.fc3=Linear(32,51*2) #ビンの数*行動数
                  
        
            def forward(self,x):
        
                h1=F.relu(self.fc1(x))
                h2=F.relu(self.fc2(h1))
                h3=self.fc3(h2)
                outputs=F.softmax(torch.reshape(h3,(-1,2,51)),dim=2)
        
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
            
            outputs=torch.reshape(self.net(data.float()),(-1,51))
            
            reshaped_targets=torch.reshape(targets,(-1,51))
            
            loss=(-reshaped_targets*((outputs+1e-8).log())).sum(dim=1).mean() #クロスエントロピー
            
            loss.backward()
            
            self.optim.step()
            
    def predict(self,x):
        
        self.net.eval()
        
        x=tensor(x,dtype=float)
        
        
        return self.net(x.float()).detach().numpy()
    
    def save(self,name=str()):
        
        torch.save(self.net.state_dict(),name)

    
def main():
    
    
    agent=categoricalDQN_agent(model=model(),
                    env=environment(),
                    n_action=2,
                    alpha=0.5,
                    g=0.9,
                    n_atoms=51,
                    min_V=-10,
                    max_V=10,
                    n_count=25,
                    using_data_rate=0.7,
                    game_over_r=[-1,1],
                    n_test=5,
                    finish_score=1,
                    save_name='categoricalDQN')
    
    agent.fit()

if __name__=='__main__':
    
    main()







