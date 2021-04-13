#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 17:44:45 2021

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
.fit(入力のnumpy配列,出力のnumpy配列)で学習
.get_weights()でモデルの重み出力
.set_weights(モデルの重み)でモデルの重み読み込み


pytorchで実装


==================================================================

class rainbowDQN_agent()でエージェントを定義する。引数はmain_modelインスタンス,target_modelインスタンス,envインスタンス,行動の数n_action,
経験再生で選ぶデータの数n_train_data,TD誤差の補正td_error_epsilon,貯めるデータん数の上限capacity,ビンの数n_atoms,行動価値の最小値min_V,最大値max_V,
学習率alpha,割引率g,1epochにまわすエピソードの数n_count,アドバンテージ学習するstep数n_step,性能評価でまわすエピソードの数n_test,
学習をおわらせるスコアfinish_score,pytorchのファイルで保存するファイル名save_name


.projected_Q（次のs’を表すNNへの入力,そのs’に到達した時に得られたr)で
r+γ＊maxQ(s',α')のQ値の分布を返す　

.game_over_projected_Q（r)でgame_over_rだった場合のQ(s,a)の分布を返す

.new_Q_distribution(next_sのarray,対応するrのarray,対応するdoneのarray)で
r+γ*Q(s',a')の分布のarrayをdoneで場合分けしながら上の.projected_Qと.game_over_projected_Q
を使って出力する


.create_train_data(epsilon,count)でn_count回エピソードが終わるまでで探索
sのarray,aのindexのarray ,rのarray,s'のarray,doneのarrayを作りDDQN,n_step,優先度つき経験再生のアルゴリズムにしたがって
教師信号の入力xと出力yを返す





.test()で今のエージェントが100%自分で行動選択した場合n_test回のエピソードで
一回あたり平均どれだけの利得を得られたかを返す

.save()でモデルを保存 モデル名は引数からのsave_name

.fit()でepochと.test()の値を表示しながら学習、test()の値がfinish_score以上になれば
学習終了、saveも毎epochしてdoubleDQNのアルゴリズムに従って重み更新



'''



import numpy as np

class rainbowDQN_agent():
    
    def __init__(self,main_model,target_model,env,n_action,n_train_data,\
                 td_error_epsilon,capacity,n_atoms,min_V,max_V,alpha,g,n_count,\
                 n_step,n_test,finish_score,save_name):
        
        print(main_model)
        
        #doubleDQN
        
        if n_train_data>capacity:
            
            print('n_train_data<capacityとなるようにしてください')
        
        self.main_model=main_model
        
        self.target_model=target_model
        
        self.env=env
        
        self.n_action=n_action
        
        self.n_train_data=n_train_data
        
        self.alpha=alpha
        
        self.g=g
        
        self.n_count=n_count
        
        self.n_test=n_test
        
        self.finish_score=finish_score
        
        self.save_name=save_name
        
        self.n_step=n_step
        
        
        #distributionalQに必要な定義
        
        self.n_atoms=n_atoms
        
        self.min_V=min_V
        
        self.max_V=max_V
        
        self.q_array=np.linspace(min_V,max_V,n_atoms)
        
        self.delta_q=(max_V-min_V)/(n_atoms-1)
        
        #memoryの定義　一連の（s,α,r,s')でindexが揃っていることに注意
        
        obs=self.env.reset()
        
        action=np.random.choice(self.n_action)
        
        obs,reward,done=self.env.step(action)
        
        self.memory_s=obs.reshape(1,-1)
        
        self.memory_a_index=np.array(action)
        
        self.memory_r=np.array(reward)
        
        self.memory_next_s=obs.reshape(1,-1)
        
        self.memory_done=np.array(int(done))
        
        self.TD_error=np.array([])
        
        self.td_error_epsilon=td_error_epsilon
        
        self.capacity=capacity
        
        
        
    def projected_Q(self,r,next_Q_dist):
        
        new_q_array=r+np.power(self.g,self.n_step)*self.q_array
        
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
    
    def new_Q_distribution(self,next_s,r,done):
        
        #DubleDQNの処理
        
        y1=self.target_model.predict(next_s)
        
        y2=self.main_model.predict(next_s)
        
        
        max_Q_index=np.argmax(np.dot(y2,self.q_array),axis=1)
            
        
        next_Q_dist=y1[np.arange(len(y1)),max_Q_index]
        
        
        new_Q_dist=[]
        
        for one_next_Q_dist,one_r,one_done in zip(next_Q_dist,r,done):
            
            if one_done: #ゲームオーバーしていない場合
                
                
            
                new_Q_dist.append(self.game_over_projected_Q(one_r))
                
            else: #ゲームオーバーしている場合
                
                new_Q_dist.append(self.projected_Q(one_r,one_next_Q_dist))
                
        new_Q_dist=np.array(new_Q_dist)
        
        
            
        return new_Q_dist
    
    
    def create_train_data(self):
        
        
        
        
        
        #n_count回分のデータ作成
        
        for i in range(self.n_count):
            
            #エピソードが終わっているかの変数doneをFalseに初期化、環境も初期化
            
            done=False
            
            observation=self.env.reset()
            
            
            
            while not done:   
                
                multi_step_reward=0
                
                self.memory_s=np.append(self.memory_s,observation.reshape(1,-1),0)
                
                y2=self.main_model.predict(observation.reshape(1,-1))[0]
                
                action=np.argmax(np.dot(y2,self.q_array))
                
                self.memory_a_index=np.append(self.memory_a_index,np.array(action))
                
                observation,reward,done=self.env.step(action)
                
                multi_step_reward+=reward
                
                if not done:
                    
                
                    for j in range(self.n_step-1):#n-step学習
                    
                        if done:
                        
                            multi_step_reward+=np.power(self.g,j-1)*reward
                        
                            break
                        
                    
                        y2=self.main_model.predict(observation.reshape(1,-1))[0]
                    
                        action=np.argmax(np.dot(y2,self.q_array))
                
                        observation,reward,done=self.env.step(action)
                    
                        multi_step_reward+=np.power(self.g,j+1)*reward
                        
                self.memory_done=np.append(self.memory_done,np.array(int(done)))
                
                self.memory_r=np.append(self.memory_r,np.array(multi_step_reward))
                
                self.memory_next_s=np.append(self.memory_next_s,observation.reshape(1,-1),0)
                
        #x、yはニューラルネットの入力と出力
        new_Q=self.main_model.predict(self.memory_s)
        
       #doubleQ-learningの更新式でyを更新
        
        new_Q[np.arange(len(self.memory_s)),self.memory_a_index]=\
        (1-self.alpha)*new_Q[np.arange(len(self.memory_s)),self.memory_a_index]+\
        self.alpha*self.new_Q_distribution(self.memory_s,self.memory_r,self.memory_done)
        
        x=self.memory_s
        
        y=new_Q
        
        #優先度付き経験再生
        
        if len(self.memory_s)>self.n_train_data:
            
            TD_error=(new_Q[np.arange(len(self.memory_s)),self.memory_a_index]*
                            np.log((self.target_model.predict(self.memory_s)\
                                      [np.arange(len(self.memory_s)),self.memory_a_index]))).sum(axis=1)+self.td_error_epsilon
            
            
            train_index=np.random.choice(len(self.memory_s),self.n_train_data,\
                                         p=TD_error/TD_error.sum(),replace=False)
            
            
            self.TD_error=TD_error
            
            x=x[train_index]
            y=y[train_index]
            
            
        return x,y
    
    
    
    def test(self):
        
        r=0
        
        for i in range(self.n_test):
            
            done=False
            
            observation=self.env.reset()
            
            while not done:
                
                                
                y2=self.main_model.predict(observation.reshape(1,-1))[0]
                
                action=np.argmax(np.dot(y2,self.q_array))
                
                observation,reward,done=self.env.step(action)
                
                r+=reward
                
            
        
        
                
        return r/self.n_test
    
    def save(self,save_name):
        
        self.main_model.save(save_name)
        
        
    def forget_memory(self):
        
        #TD誤差が小さいデータを優先的に忘れる
        
        if len(self.memory_s)>self.capacity:
            
            print(len(self.TD_error),len(self.memory_s))
            
            
            inverse_TD_error=1/(self.TD_error+1e-8)
            
            forget_index=np.random.choice(len(self.memory_s),len(self.memory_s)-self.capacity,\
                                         p=inverse_TD_error/inverse_TD_error.sum(),replace=False)
            
            self.memory_s=np.delete(self.memory_s,forget_index,axis=0)
            self.memory_a_index=np.delete(self.memory_a_index,forget_index,axis=0)
            self.memory_r=np.delete(self.memory_r,forget_index,axis=0)
            self.memory_next_s=np.delete(self.memory_next_s,forget_index,axis=0)
            self.memory_done=np.delete(self.memory_done,forget_index,axis=0)
            
           
    
    def fit(self):
        
        epoch=1
        
        score=self.finish_score-1
        
        while  self.finish_score>score: #学習の継続条件
            
            
            #教師データ取得
            x,y=self.create_train_data()
            
            #main_modelの重みをtarget_modelの重みに共有
            
            self.target_model.set_weights(self.main_model.get_weights())
           
            #main_model学習
            self.main_model.fit(x,y)
            
            #データ量がcapacityを超えてる場合は忘れる
            self.forget_memory()
            
            #モデル保存
            self.save(self.save_name)
            
            #現在のモデルの平均スコア取得
            score=self.test()
            
            print('epoch:'+str(epoch)+'  score:'+str(score))
            
            epoch+=1
            
        print('fitting finished')
        
        
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
from torch.nn import functional as F,Module,Parameter
from torch.optim import Adam


class NoisyLinear(Module):
    
    def __init__(self,n_in,n_out,sig_0=0.5):
        
        super(NoisyLinear,self).__init__()
        
        self.n_in=n_in
        self.n_out=n_out
        
        mu_const=1/np.sqrt(self.n_in) #論文よりμは一様分布 U(-1/√n_in,1/√n_in)で初期化
        sig_const=sig_0/np.sqrt(self.n_in)#論文よりσ^2は 0.5/√n_inで初期化
        
        self.mu_w=Parameter(torch.Tensor(n_out,n_in))
        self.sig_w=Parameter(torch.Tensor(n_out,n_in))
        
        self.mu_b=Parameter(torch.Tensor(n_out))
        self.sig_b=Parameter(torch.Tensor(n_out))
        
        self.mu_w.data=tensor(np.random.rand(n_out,n_in)*2*mu_const-mu_const)
        self.mu_b.data=tensor(np.random.rand(n_out)*2*mu_const-mu_const)
        
        self.sig_w.data=tensor(np.ones((n_out,n_in))*sig_const)
        self.sig_b.data=tensor(np.ones(n_out)*sig_const)
        
    def f_(self,x):
        
        return torch.sign(x)*torch.sqrt(torch.abs(x))
    
    def forward(self,x):
        
        epsilon_i=self.f_(torch.randn(1,self.n_in))
        epsilon_j=self.f_(torch.randn(self.n_out,1))
        
        #factrised gaussian noiseでノイズ生成
        
        epsilon_w=torch.matmul(epsilon_j,epsilon_i)
        epsilon_b=epsilon_j.squeeze()
        
        w=self.mu_w+self.sig_w*epsilon_w
        b=self.mu_b+self.sig_b*epsilon_b
        
        
        
        
        return F.linear(x.double(),w.double(),b.double())


        
class model():
    
    def __init__(self):
        
        
        class Net(Module):
            
            def __init__(self):
                      
                super(Net,self).__init__()
                self.nl1=NoisyLinear(4,16)
                self.nl2=NoisyLinear(16,32)
                self.nl_V=NoisyLinear(32,51) 
                self.nl_adv=NoisyLinear(32,51*2)#ビンの数*行動数
                  
        
            def forward(self,x):
        
                h1=F.relu(self.nl1(x))
                h2=F.relu(self.nl2(h1))
                
                adv=self.nl_adv(h2).reshape(-1,2,51)
        
                V=self.nl_V(h2).expand(2,-1,51).transpose(1,0)
                
                adv_mean=adv.mean(axis=1).expand(2,-1,51).transpose(1,0).detach()
                
                outputs=V+adv-adv_mean
                
                
                '''
                adv.mean(1）はshapeが（N)なのでkeepdim=Trueで（N,1)にして
                V+advと足し算できるように.expand(-1,行動の数=2)
                しています。
                '''
                
                outputs=F.softmax(outputs,dim=2)
        
                return outputs
        
        
       
        self.net=Net()
        
        self.optim=Adam(self.net.parameters(),lr=0.01)
       
        
    def fit(self,x,y):
        
        x=tensor(x,dtype=float)
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
        
    def get_weights(self):
        
        return self.net.state_dict()
    
    def set_weights(self,model_weights):
        
        self.net.load_state_dict(model_weights)

    
def main():
    
    
    agent=rainbowDQN_agent(main_model=model(),
                           target_model=model(),
                           env=environment(),
                           n_action=2,
                           n_train_data=10000,
                           td_error_epsilon=0.0001,
                           capacity=100000,
                           n_atoms=51,
                           min_V=-10,
                           max_V=10,
                           alpha=0.3,
                           g=0.9,
                           n_count=75,
                           n_step=3,
                           n_test=5,
                           finish_score=200,
                           save_name='rainbowDQN')
   
    agent.fit()




if __name__=='__main__':
    
    main()











































