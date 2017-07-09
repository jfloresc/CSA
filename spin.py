#!/usr/bin/env python

import numpy as np
import copy 
import json

# it reproduces results from
# Seung-Yeon Kim, Sung Jong Lee, and Jooyoung Lee , Ground-state energy and energy landscape of the Sherrington-Kirkpatrick spin glass, Phys.Rev.B, Vol. 76, 184412-1 - 184412-7 (2007).
# written in 07/08/2015 by jose flores-canales

def runmain(instances,icall):

  np.random.seed()

  N = 511 
 
  J=np.random.choice([-1,1],size=(N,N))
  nameJ = 'J'+str(icall)+'.dat'
  np.savetxt(nameJ,J,fmt='%d')

  #np.random.seed()
  #spins = initialize(N)
  #print 'Energy', hamiltonian(spins,J,N)
  #spins_min, ene_min = minimize(spins,J,N)
  #print 'Min. Energy', ene_min
  
  #return  0 

  init_bank = 50 
  addconfig = init_bank 
  nseed = 10
  nmax_trials = 10000
  nrounds = 27 

  k = np.power(0.4,1./nmax_trials)
  
  filename = 'output'+str(icall)+'.dat'
  global_ene = 'minimum'+str(icall)+'.dat'
  print "Opening files: ", filename, global_ene
  f = open(filename,'w')
  ef = open(global_ene,'w')

  istep  = 0

  while istep < instances:
    nbank = init_bank 
    initglobalvar()

    bank1,ene_max1,i_max = createbank(nbank,J,N)
    name = 'bank1_'+str(icall)+'_'+str(istep)+'.dat'
    with open(name,'w') as f1:
      json.dump(bank1,f1)
      f1.flush()
    bank = copy.deepcopy(bank1)
    bank_max = i_max
    ene_max  = ene_max1

    dave = average_dist(bank1,nbank,N)
    dcut = dave*0.5
    print "nbank: ", nbank
    print "nseed: ", nseed
    print "dcut: ", dcut
    print "ratio: ", k

    iround = 0
    print 'Iround', iround
    bank_ene_min, bank_i_min = findminimum(bank,nbank,N)
    bank1_ene_min, bank1_i_min = findminimum(bank1,nbank,N)
    print 'i_bank_max', bank_max,'Max. Ene_max',ene_max,'Min. Ene_min',bank_ene_min,'i_bank_min',bank_i_min,'nbank',nbank,'i_bank1_max',i_max,'Ene_max1',ene_max1,'Ene_min1',bank1_ene_min,'i_bank1_min',bank1_i_min

    while iround < nrounds:
      if iround % 3  == 0 and iround > 1:
        print 'Iround', iround
        t,ene_max_t,i_max_t = createbank(addconfig,J,N)
      #print 'after adding config. to bank, ene_max', ene_max,'ene_max_new', ene_max_t
      # increase bank1 and bank
        if (ene_max_t > ene_max):
          bank_max = i_max_t + len(bank)
        #bank_max = i_max
          ene_max = ene_max_t
        if (ene_max_t > ene_max1):
          i_max = i_max_t + len(bank1)
          ene_max1 = ene_max_t
        bank1 = bank1 + t
        bank = bank + t 
        nbank += addconfig 
        dcut = dave*0.5
        bank_ene_min, bank_i_min = findminimum(bank,nbank,N)
        bank1_ene_min, bank1_i_min = findminimum(bank1,nbank,N)
        print 'i_bank_max', bank_max,'Max. Ene_max',ene_max,'Min. Ene_min',bank_ene_min,'i_bank_min',bank_i_min,'nbank',nbank,'i_bank1_max',i_max,'Ene_max1',ene_max1,'Ene_min1',bank1_ene_min,'i_bank1_min',bank1_i_min

      record = [False for i in xrange(len(bank1))]
      if iround % 3 == 0 and iround > 1:
        nupdates = xrange(addconfig/nseed)
      else:
        nupdates = xrange(len(bank1)/nseed)
     
      for iupdate in nupdates:  
        genes_ene,record = generateconfig(bank1,bank,nbank,nseed,record,J,N,iround,addconfig)
        f.write('%6d %5.3f\n' % (getglobalvar(),dcut))
        for i in xrange(nseed):
          bank,ene_max = updatebank(genes_ene[i],nseed,bank,nbank,bank_max,dcut,J,N)
          #trials += 1
          if getglobalvar() < nmax_trials: 
            dcut = dcut*k
          else:
            dcut = dave*0.2
    #print 'after update, ene_max', ene_max
      #print 'trials, dcut: ',trials, dcut
      iround += 1
    bank_ene_min,bank_i_min = findminimum(bank,nbank,N)
    print "min from bank", bank_ene_min 
    ef.write('%6d %5.6f\n' % (istep,bank_ene_min))
    name = 'bank_min'+str(icall)+'_'+str(istep)+'.dat'
    with open(name,'w') as f2:
      json.dump(bank,f2)
      f2.flush()
    del bank, bank1
    istep +=1
  # closing files
  f.close()
  ef.close()

def findminimum(bank,nbank,N):
  ene_min = 10*N

  for i in xrange(nbank):
    if bank[i][1] < ene_min:
      ene_min = bank[i][1]
      i_min = i

  return ene_min, i_min

def projection(ta,tb,N):
  q = 0
  for i,j in zip(ta,tb):
    q += i*j
  return abs(q)/float(N)

def updatebank(genes_ene,nseed,bank,nbank,i_max,dcut,J,N):
  ene_max = bank[i_max][1]
  dij_min = N*100
  for j in xrange(nbank):
    dij = distance(genes_ene[0],bank[j][0],N)
    if dij < dij_min: 
      dij_min = dij
      j_min = j

  if dij_min < dcut: 
    if genes_ene[1] < bank[j_min][1]:
      bank[j_min][0] = genes_ene[0]
      bank[j_min][1] = genes_ene[1]
  elif dij_min > dcut:
    if genes_ene[1] < ene_max:
      #print 'update ene_max', ene_max, 'new_ene:', genes_ene[1]
      bank[i_max][0] = genes_ene[0]
      bank[i_max][1] = genes_ene[1]
      ene_max = genes_ene[1]

  return bank,ene_max

def generateconfig(bank1,bank,nbank,nseed,record,J,N,iround,addconfig):
  c = 0
  genes_ene = [] 
  #p = np.random.randint(nseed)
  if iround % 3 == 0 and iround > 1:
    configs = xrange(nbank-addconfig,nbank)
  else:
    configs = xrange(nbank)
  nrand_bank = 1 
  randbank,rand_ene_max1,rand_i_max = createbank(nrand_bank,J,N)
  for p in configs:
    if c == nseed: break
    if not record[p]:
      gene1 = copy.deepcopy(bank1[p][0])
      record[p] = True
      for i in xrange(nrand_bank):
        gene2 = copy.deepcopy(randbank[i][0])
        gene1 = merge(gene1,gene2,N)   
      gene3, ene_min = minimize(gene1,J,N)
      #print ene_min
      genes_ene.append([gene3,ene_min])
      c += 1 

  return genes_ene,record

def merge(gene1,gene2,N):
# maximum number of spins to be replacement at a time
  b = 4 
# number of replacements  
  nrepl = 10 
  for i in xrange(nrepl):
    #if np.random.random()<0.9:
    #block = np.random.randint(b)+1
    block = b
    p = np.random.randint(N-block+1)
    gene1[p:p+block]=gene2[p:p+block]
    #else:
    #  block = b 
    #  p = np.random.randint(N-block+1)
    #  gene1[p:p+block]=gene2[p:p+block]

  return gene1

def createbank(nbank,J,N):
  b = []
  ene_max = -100*N
  for i in xrange(nbank):
    t = initialize(N)
    t_min, ene_min = minimize(t,J,N)
    b.append([t_min,ene_min])
    if ene_min > ene_max:
      ene_max = ene_min
      imax = i
  return b, ene_max, imax

def average_dist(bank,nbank,N):
  c,d = 0,0
  for i in xrange(nbank):
    for j in xrange(i+1,nbank):
      d += distance(bank[i][0],bank[j][0],N)
      c += 1

  return float(d)/c

def distance(ta,tb,N):
  d = 0
  for i,j in zip(ta,tb):
    if i!=j: d += 1

  return min(d,N-d)

def minimize(spins,J,N):
  trial = getglobalvar() 
  setglobalvar(trial+1)
  init = hamiltonian(spins,J,N)
  t = copy.deepcopy(spins)
  c = 0
  while c < N: 
    for i in xrange(N):
      dene = dhamiltonian(t,J,N,i)
      if dene < 0:
        t[i] = -t[i]
        c = 0
      else:
        c += 1
      if c == N:
        break
  ene = hamiltonian(t,J,N)

  return t, ene 

def hamiltonian(spins,J,N):
  H = 0

  for i in xrange(N):
    for j in xrange(i+1,N):
      H += spins[i]*spins[j]*J[i,j]
  return  -float(H)/(np.sqrt(N)*N)

def dhamiltonian(spins,J,N,idx):
  E_old = 0
  E_new = 0

# same hamiltonian, i,j != idx
#  for i in xrange(N):
#    if i == idx: continue
#    for j in xrange(i+1,N):
#      if j == idx: continue
#      E += spins[i]*spins[j]*J[i,j]

# changes for i = idx
  s = spins[idx]
  for j in xrange(idx+1,N):
    E_old += s*spins[j]*J[idx,j]
    E_new += -s*spins[j]*J[idx,j]
    
# changes for j = idx
  for i in xrange(idx):
    E_old += spins[i]*s*J[i,idx]
    E_new += -spins[i]*s*J[i,idx]

  return  float(-E_new+E_old)/(np.sqrt(N)*N)

def initialize(N):
  vector = []
  J = lambda p: -1 if p == 0 else 1
  for i in xrange(N):
    p = np.random.randint(2)
    vector.append(J(p))
  return vector

def initglobalvar():
  global __trials__
  __trials__ = 0

def getglobalvar():
  global __trials__
  return __trials__

def setglobalvar(t):
  global __trials__
  __trials__ = t

if ( __name__ == '__main__' ):
  
  __trials__ = 0

  for i in xrange(10):
    runmain(10,i)
