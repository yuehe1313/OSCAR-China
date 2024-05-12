
# # -----------------------------------------------------------------------------------------------

# f--forest 
# c--cropland
# p--pasture 
# g--non-forest
# ignore urban area

# NetLUCC3: f vs c vs pg
# NetLUCC4: f vs c vs p vs g

# assume：minimum transition rates 
# assume：Expanded Pasture and Non-forest areas are taken in proportion from Forest and Cropland using the ratio of changes in forest to cropland area that year, and vice versa.

# LUC1 = forest in current year - forest in next year
# LUC2 = cropland in current year - cropland in next year
# LUC3 = pasture in current year - pasture in next year
# LUC4 = non-forest in current year - non-forest in next year
# LUC34 = pasture and non-forest in current year - pasture and non-forest in next year

# LUC_type=1: LUC1>0, LUC2<0, LUC3<0,  
# LUC_type=10: LUC1<0, LUC2>0, LUC3<0,  
# LUC_type=11: LUC1>0, LUC2>0, LUC3<0, 
# LUC_type=100: LUC1<0, LUC2<0, LUC3>0, 
# LUC_type=101: LUC1>0, LUC2<0, LUC3>0, 
# LUC_type=110: LUC1<0, LUC2>0, LUC3>0, 

# LUC_type34=0: LUC3<0, LUC4<0, 
# LUC_type34=1: LUC3>0, LUC4<0, 
# LUC_type34=10: LUC3<0, LUC4>0,
# LUC_type34=11: LUC3>0, LUC4>0, 

#-----------LUC_type=1, LUC_type34=0,f_d,c_u,pg_u,p_u,g_u---------
# f2c=-LUC2,f2pg=-LUC34,f2p=-LUC3,f2g=-LUC4
#-----------LUC_type=1, LUC_type34=1,f_d,c_u,pg_u,p_d,g_u---------
# f2c=-LUC2,f2pg=-LUC34,f2g=-LUC4-LUC3=LUC1+LUC2,p2g=LUC3
#-----------LUC_type=1, LUC_type34=10,f_d,c_u,pg_u,p_u,g_d ---------
# f2c=-LUC2,f2pg=-LUC34,f2p=-LUC4-LUC3=LUC1+LUC2,g2p=LUC4

#-----------LUC_type=10, LUC_type34=0,f_u,c_d,pg_u,p_u,g_u ---------
# c2f=-LUC1,c2pg=-LUC34,c2p=-LUC3,c2g=-LUC4
#-----------LUC_type=10, LUC_type34=1,f_u,c_d,pg_u,p_d,g_u---------
# c2f=-LUC1,c2pg=-LUC34,c2g=LUC1+LUC2,p2g=LUC3
#-----------LUC_type=10, LUC_type34=10,f_u,c_d,pg_u,p_u,g_d---------
# c2f=-LUC1,c2pg=-LUC34,c2p=LUC1+LUC2,g2p=LUC4

#-----------LUC_type=11, LUC_type34=0,f_d,c_d,pg_u,p_u,g_u---------
# f2pg=LUC1,c2pg=LUC2
# f2p=LUC1*(-LUC3/(-LUC3-LUC4))
# f2g=LUC1*(-LUC4/(-LUC3-LUC4))
# c2p=LUC2*(-LUC3/(-LUC3-LUC4))
# c2g=LUC2*(-LUC4/(-LUC3-LUC4))
#-----------LUC_type=11, LUC_type34=1,f_d,c_d,pg_u,p_d,g_u---------
# f2pg=f2g=LUC1,c2pg=c2g=LUC2,p2g=LUC3
#-----------LUC_type=11, LUC_type34=10,f_d,c_d,pg_u,p_u,g_d---------
# f2pg=f2p=LUC1,c2pg=c2p=LUC2,g2p=LUC4

#-----------LUC_type=100, LUC_type34=1,f_u,c_u,pg_d,p_d,g_u---------
# pg2f=p2f=-LUC1,pg2c=p2c=-LUC2,p2g=-LUC4
#-----------LUC_type=100, LUC_type34=10,f_u,c_u,pg_d,p_u,g_d---------
# pg2f=g2f=-LUC1,pg2c=g2c=-LUC2,g2p=-LUC3
#-----------LUC_type=100, LUC_type34=11,f_u,c_u,pg_d,p_d,g_d---------
# pg2f=-LUC1,pg2c=-LUC2
# p2f=LUC3*(-LUC1/(-LUC1-LUC2))
# p2c=LUC3*(-LUC2/(-LUC1-LUC2))
# g2f=LUC4*(-LUC1/(-LUC1-LUC2))
# g2c=LUC4*(-LUC2/(-LUC1-LUC2))

#-----------LUC_type=101, LUC_type34=1,f_d,c_u,pg_d,p_d,g_u---------
# f2c=LUC1,pg2c=LUC34,p2c=-LUC1-LUC2=LUC34,p2g=-LUC4
#-----------LUC_type=101, LUC_type34=10,f_d,c_u,pg_d,p_u,g_d---------
# f2c=LUC1,pg2c=LUC34,g2c=-LUC1-LUC2=LUC34,g2p=-LUC3
#-----------LUC_type=101, LUC_type34=11,f_d,c_u,pg_d,p_d,g_d---------
# f2c=LUC1,pg2c=LUC34,p2c=LUC3,g2c=LUC4

#-----------LUC_type=110, LUC_type34=1,f_u,c_d,pg_d,p_d,g_u---------
# c2f=LUC2,pg2f=LUC34,p2f=-LUC1-LUC2=LUC34,p2g=-LUC4
#-----------LUC_type=110, LUC_type34=10,f_u,c_d,pg_d,p_u,g_d---------
# c2f=LUC2,pg2f=LUC34,g2f=-LUC1-LUC2=LUC34,g2p=-LUC3
#-----------LUC_type=110, LUC_type34=11,f_u,c_d,pg_d,p_d,g_d---------
# c2f=LUC2,pg2f=LUC34,p2f=LUC3,g2f=LUC4

#-------LUC_tpye=0,LUC_type34=1,p<=>g
# p2g=LUC3
#-------LUC_tpye=0,LUC_type34=10,p<=>g
# g2p=LUC4

# # -----------------------------------------------------------------------------------------------
import numpy as np
     
def LUC_mask(LUC_mask0,LUC_num):
    mask = np.copy(LUC_mask0)
    mask[mask!=LUC_num]=-999
    mask[mask==LUC_num]=1
    mask[mask!=1]=0
    return mask

def NetLUCC3(LUC1,LUC2,LUC3,LUC_type_p=False): 
    if np.max((LUC1+LUC2+LUC3)<1.e-4):
      lat = LUC1.shape[0]
      lon = LUC1.shape[1]
      LUC_type = np.zeros((lat,lon))
      LUCMatrix = np.zeros((3,3,lat,lon))
      LUC_type[LUC1>0] += 1 
      LUC_type[LUC2>0] += 10 
      LUC_type[LUC3>0] += 100 

      #LUC1->LUC2 & LUC1->LUC3
      LUCMatrix[0,1,LUC_type==1] = -LUC2[LUC_type==1]
      LUCMatrix[0,2,LUC_type==1] = -LUC3[LUC_type==1]
      #LUC2->LUC1 & LUC2->LUC3
      LUCMatrix[1,0,LUC_type==10] = -LUC1[LUC_type==10]
      LUCMatrix[1,2,LUC_type==10] = -LUC3[LUC_type==10] 
      #LUC3->LUC1 & LUC3->LUC2
      LUCMatrix[2,0,LUC_type==100] = -LUC1[LUC_type==100]
      LUCMatrix[2,1,LUC_type==100] = -LUC2[LUC_type==100] 
      #LUC1->LUC3 & LUC2->LUC3
      LUCMatrix[0,2,LUC_type==11] = LUC1[LUC_type==11] 
      LUCMatrix[1,2,LUC_type==11] = LUC2[LUC_type==11]
      #LUC1->LUC2 & LUC3->LUC2
      LUCMatrix[0,1,LUC_type==101] = LUC1[LUC_type==101]
      LUCMatrix[2,1,LUC_type==101] = LUC3[LUC_type==101]
      #LUC2->LUC1 & LUC3->LUC1
      LUCMatrix[1,0,LUC_type==110] = LUC2[LUC_type==110]
      LUCMatrix[2,0,LUC_type==110] = LUC3[LUC_type==110] 
      if LUC_type_p:
        return LUC_type,LUCMatrix
      else:
        return LUCMatrix
    else:
      print('error:LUC1+LUC2+LUC3 should equal zero!')
      
def NetLUCC4(LUC1,LUC2,LUC3,LUC4):
    if np.max(LUC1+LUC2+LUC3+LUC4)<1.e-4:
        lat = LUC1.shape[0]
        lon = LUC1.shape[1]
        
        LUCMatrix = np.zeros((4,4,lat,lon))
        LUC_type,LUCMatrix3 = NetLUCC3(LUC1,LUC2,LUC3+LUC4,LUC_type_p=True)
        LUCMatrix[0:2,0:2,:,:] = LUCMatrix3[0:2,0:2,:,:]
        
        LUC_type34 = np.zeros((lat,lon))
        LUC_type34[LUC3>0]+=1 
        LUC_type34[LUC4>0]+=10 
        
        LUC1_2 = LUC1+LUC2 
        mask = LUC_mask(LUC_type34,0)
        LUCMatrix[0,2,LUC_type*mask==1]=-LUC3[LUC_type*mask==1]
        LUCMatrix[0,3,LUC_type*mask==1]=-LUC4[LUC_type*mask==1]
        
        mask = LUC_mask(LUC_type34,1)
        LUCMatrix[0,3,LUC_type*mask==1]=LUC1_2[LUC_type*mask==1]
        LUCMatrix[2,3,LUC_type*mask==1]=LUC3[LUC_type*mask==1] 
            
        mask = LUC_mask(LUC_type34,10)
        LUCMatrix[0,2,LUC_type*mask==1]=LUC1_2[LUC_type*mask==1]
        LUCMatrix[3,2,LUC_type*mask==1]=LUC4[LUC_type*mask==1]
        
        mask = LUC_mask(LUC_type34,0)
        LUCMatrix[1,2,LUC_type*mask==10]=-LUC3[LUC_type*mask==10]
        LUCMatrix[1,3,LUC_type*mask==10]=-LUC4[LUC_type*mask==10]
            
        mask = LUC_mask(LUC_type34,1)
        LUCMatrix[1,3,LUC_type*mask==10]=LUC1_2[LUC_type*mask==10]
        LUCMatrix[2,3,LUC_type*mask==10]=LUC3[LUC_type*mask==10]   
            
        mask = LUC_mask(LUC_type34,10)
        LUCMatrix[1,2,LUC_type*mask==10]=LUC1_2[LUC_type*mask==10]
        LUCMatrix[3,2,LUC_type*mask==10]=LUC4[LUC_type*mask==10] 

        mask = LUC_mask(LUC_type34,1)
        LUCMatrix[2,0,LUC_type*mask==100]=-LUC1[LUC_type*mask==100]
        LUCMatrix[2,1,LUC_type*mask==100]=-LUC2[LUC_type*mask==100]
        LUCMatrix[2,3,LUC_type*mask==100]=-LUC4[LUC_type*mask==100]
        
        mask = LUC_mask(LUC_type34,10)
        LUCMatrix[3,0,LUC_type*mask==100]=-LUC1[LUC_type*mask==100]
        LUCMatrix[3,1,LUC_type*mask==100]=-LUC2[LUC_type*mask==100]
        LUCMatrix[3,2,LUC_type*mask==100]=-LUC3[LUC_type*mask==100]
            
        mask = LUC_mask(LUC_type34,11)
        LUCMatrix[2,0,LUC_type*mask==100]=LUC3[LUC_type*mask==100]*\
            (-LUC1)[LUC_type*mask==100]/(-LUC2-LUC1)[LUC_type*mask==100]
        LUCMatrix[2,1,LUC_type*mask==100]=LUC3[LUC_type*mask==100]*\
            (-LUC2)[LUC_type*mask==100]/(-LUC2-LUC1)[LUC_type*mask==100]
        LUCMatrix[3,0,LUC_type*mask==100]=LUC4[LUC_type*mask==100]*\
            (-LUC1)[LUC_type*mask==100]/(-LUC2-LUC1)[LUC_type*mask==100]
        LUCMatrix[3,1,LUC_type*mask==100]=LUC4[LUC_type*mask==100]*\
            (-LUC2)[LUC_type*mask==100]/(-LUC2-LUC1)[LUC_type*mask==100]

        mask = LUC_mask(LUC_type34,0)
        LUCMatrix[0,2,LUC_type*mask==11]=LUC1[LUC_type*mask==11]*\
            (-LUC3)[LUC_type*mask==11]/(-LUC3-LUC4)[LUC_type*mask==11]
        LUCMatrix[0,3,LUC_type*mask==11]=LUC1[LUC_type*mask==11]*\
            (-LUC4)[LUC_type*mask==11]/(-LUC3-LUC4)[LUC_type*mask==11]
        LUCMatrix[1,2,LUC_type*mask==11]=LUC2[LUC_type*mask==11]*\
            (-LUC3)[LUC_type*mask==11]/(-LUC3-LUC4)[LUC_type*mask==11]
        LUCMatrix[1,3,LUC_type*mask==11]=LUC2[LUC_type*mask==11]*\
            (-LUC4)[LUC_type*mask==11]/(-LUC3-LUC4)[LUC_type*mask==11]

        mask = LUC_mask(LUC_type34,1)
        LUCMatrix[0,3,LUC_type*mask==11]=LUC1[LUC_type*mask==11]
        LUCMatrix[1,3,LUC_type*mask==11]=LUC2[LUC_type*mask==11]
        LUCMatrix[2,3,LUC_type*mask==11]=LUC3[LUC_type*mask==11]
            
        mask = LUC_mask(LUC_type34,10)
        LUCMatrix[0,2,LUC_type*mask==11]=LUC1[LUC_type*mask==11]
        LUCMatrix[1,2,LUC_type*mask==11]=LUC2[LUC_type*mask==11]
        LUCMatrix[3,2,LUC_type*mask==11]=LUC4[LUC_type*mask==11]
        
        mask = LUC_mask(LUC_type34,1)
        LUCMatrix[2,1,LUC_type*mask==101]=-LUC1_2[LUC_type*mask==101]
        LUCMatrix[2,3,LUC_type*mask==101]=-LUC4[LUC_type*mask==101]
        
        mask = LUC_mask(LUC_type34,10)
        LUCMatrix[3,1,LUC_type*mask==101]=-LUC1_2[LUC_type*mask==101]
        LUCMatrix[3,2,LUC_type*mask==101]=-LUC3[LUC_type*mask==101]
            
        mask = LUC_mask(LUC_type34,11)
        LUCMatrix[2,1,LUC_type*mask==101]=LUC3[LUC_type*mask==101]
        LUCMatrix[3,1,LUC_type*mask==101]=LUC4[LUC_type*mask==101]

        mask = LUC_mask(LUC_type34,1)
        LUCMatrix[2,0,LUC_type*mask==110]=-LUC1_2[LUC_type*mask==110]
        LUCMatrix[2,3,LUC_type*mask==110]=-LUC4[LUC_type*mask==110]
        
        mask = LUC_mask(LUC_type34,10)
        LUCMatrix[3,0,LUC_type*mask==110]=-LUC1_2[LUC_type*mask==110]
        LUCMatrix[3,2,LUC_type*mask==110]=-LUC3[LUC_type*mask==110]
            
        mask = LUC_mask(LUC_type34,11)
        LUCMatrix[2,0,LUC_type*mask==110]=LUC3[LUC_type*mask==110]
        LUCMatrix[3,0,LUC_type*mask==110]=LUC4[LUC_type*mask==110]
        
        mask = LUC_mask(LUC_type34,1)
        LUC_type = LUC_mask(LUC_type,0)
        LUCMatrix[2,3,LUC_type*mask==1]=LUC3[LUC_type*mask==1]

        mask = LUC_mask(LUC_type34,10)
        LUCMatrix[3,2,LUC_type*mask==1]=LUC4[LUC_type*mask==1]
        
        LUCMatrix[np.isnan(LUCMatrix)]=0
        return LUCMatrix
    else:
      print('error:LUC1+LUC2+LUC3+LUC4 should equal zero!')



