# This file is part of the master thesis "Variational crimes in the Localized orthogonal decomposition method":
#   https://github.com/TiKeil/Masterthesis-LOD.git
# Copyright holder: Tim Keil 
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

# TODO: This class needs a clean up !

import numpy as np
import random

class Coefficient2d:
    def __init__(self,NWorldFine, 
                    bg=0.01, 
                    val=1, 
                    length=1, 
                    thick=1, 
                    space=1, 
                    probfactor = 10, 
                    right=0, 
                    down=0, 
                    diagr1=0, 
                    diagr2=0, 
                    diagl1=0, 
                    diagl2=0, 
                    LenSwitch=None, 
                    thickSwitch=None, 
                    equidistant=None, 
                    ChannelHorizontal=None, 
                    ChannelVertical=None,
                    BoundarySpace=None,
                    Boxes2n=None,
                    Channels2n=None,
                    NewShapes=None,
                    RandomInverse=None,
                    TestExample=None):
        
        '''
        2dCoefficient   
        '''

        self.NWorldFine = NWorldFine 
        self.bg = bg 
        self.val = val 
        
        #basic properties
        self.length = length 
        self.thick = thick 
        self.space = space 
        
        #probability
        self.probfactor = probfactor
        
        #shapes
        self.right = right 
        self.down = down 
        self.diagr1 = diagr1 
        self.diagr2 = diagr2 
        self.diagl1 = diagl1 
        self.diagl2 = diagl2 
        
        #more features
        self.LenSwitch = LenSwitch 
        self.thickSwitch = thickSwitch 
        self.equidistant = equidistant 
        self.ChannelHorizontal = ChannelHorizontal 
        self.ChannelVertical = ChannelVertical
        self.BoundarySpace = BoundarySpace
        self.Boxes2n = Boxes2n
        self.Channels2n = Channels2n
        
        #additional memory
        self.Matrix = None
        self.RandomMatrix = None
        
        #the shaperemember has 1 := 7
        self.ShapeRemember = None
        self.ShapeRememberOriginal = None
        
        #channelsafer
        self.Channelsafer = None
        self.nomore = 0
        
        self.NewShapes = NewShapes
        #newshapes
        self.Shapes = 0
        self.ShapeMatrixes = np.array([])
        self.ShapeSizes = np.array([])
        self.ShapeIndex = [0]
        self.ShapeCoords = []
        self.CoordsIndex = [0]
        
        #valuecount
        self.valuecounter = None
        
        self.TestExample = TestExample
        
    def NewShape(self, ShapeBuildMatrix):
        '''
        you need to enumerate the new shapes by yourself
        '''
        
        self.Shapes += 1
        Sizing = np.shape(ShapeBuildMatrix)
                
        self.ShapeIndex.append(Sizing[0]*Sizing[1]+self.ShapeIndex[self.Shapes-1])
        Shaping = ShapeBuildMatrix.flatten()
        self.ShapeMatrixes = np.append(self.ShapeMatrixes,Shaping)
        Shapesizes = self.ShapeSizes.flatten() 
        Shapesizes= np.append(Shapesizes,Sizing)
        self.ShapeSizes = np.reshape(Shapesizes,(self.Shapes,2))
        
        #search indizes
        NumberOfShapes = Sizing[0]
        
        LengthPosition = []
        ThickPosition = []
        ShapeCoords = []
        ShapeCoords.extend([0,0])
        
        for shape in range(0,NumberOfShapes):
            #put them together uniquely with the following properties
            ShapeIndex = ShapeBuildMatrix[shape][0]                 #what shape
            ShapeLength = ShapeBuildMatrix[shape][1]                #what length
            ShapeThick = ShapeBuildMatrix[shape][2]                 #what thick
            ShapeShapes = ShapeBuildMatrix[shape][4]
            ShapeInverser = ShapeBuildMatrix[shape][3]                #how many shapes 
            
            pxs = ShapeCoords[shape*2]
            pys = ShapeCoords[shape*2+1]
            
            for i in range(0,ShapeShapes):
                LengthPosition = ShapeBuildMatrix[shape][5+2*i]             #where regarding length
                ThickPosition = ShapeBuildMatrix[shape][6+2*i]              
                px, py = self.SearchShapeIndex(ShapeIndex, LengthPosition, ThickPosition)
                px *= ShapeInverser
                py *= ShapeInverser
                ShapeCoords.extend([pxs+px,pys+py])
            
        self.ShapeCoords.extend(ShapeCoords)
        summand = self.CoordsIndex[self.Shapes-1]
        self.CoordsIndex.append(summand + 2*NumberOfShapes)
        
        return 0
        
        ########################### IndexSearch ###################################

    def SearchShapeIndex(self, Shape, LengthPosition, ThickPosition):
        if Shape == 1:
            px = ThickPosition
            py = LengthPosition
                
        elif Shape == 2:
            px = ThickPosition+LengthPosition
            py = LengthPosition
                        
        elif Shape == 3:
            px = LengthPosition
            py = LengthPosition+ThickPosition
                        
        elif Shape == 4:
            px = LengthPosition
            py = ThickPosition
                        
        elif Shape ==5:
            px = ThickPosition+LengthPosition
            py = -LengthPosition
                    
        elif Shape == 6:
            px = LengthPosition
            py = -LengthPosition+ThickPosition
        
        return px, py            
    
        
        ############### BUILD FUNCTION #################    
    def BuildCoefficient(self):
        #random seed
        random.seed(20)
        
        #regain properties
        NWorldFine = self.NWorldFine 
        bg = self.bg  
        val = self.val 
        
        #basic properties
        thick = self.thick  
        space = self.space  
        
                
        #more features
        LenSwitch = self.LenSwitch  
        thickSwitch = self.thickSwitch  
        equidistant = self.equidistant  
        ChannelHorizontal = self.ChannelHorizontal  
        ChannelVertical = self.ChannelVertical 
        BoundarySpace = self.BoundarySpace 
        
        #essential
        A = np.zeros(NWorldFine)
        B = A.copy()
        #initial for shape remember
        S = np.array([])
        #for remember
        shapecounter = 0
        
        #add not included coefficients 
        #Maybe improve 
        if self.Channels2n:
            self.down = True
            self.length = NWorldFine[0]
            self.thick = 1
            self.space = 1
            #EvenChannels
            Abase = np.zeros(NWorldFine[1])
            for i in range(2,NWorldFine[1]/2-1,2):
                shapecounter += 1
                S = np.append(S,[4,self.length,1])
                Abase[i] = val-bg
                Abase[i+NWorldFine[1]/2-1] = val-bg
    
            AbaseCube = np.tile(Abase[...,np.newaxis], [NWorldFine[0],1])
            AbaseCube = AbaseCube[...,np.newaxis]
            ABase = AbaseCube.flatten()
            A = ABase.reshape(NWorldFine)
            A += bg
            S = np.reshape(S,(shapecounter,3))
            self.ShapeRemember = S
            self.ShapeRememberOriginal = S
            self.Matrix = A
            self.RandomMatrix = A
            return A
            
        if self.Boxes2n:
            self.right = True
            self.length = 1
            self.thick = 1
            self.space = 1
            #annas coeff for even boxes
            A = np.zeros(NWorldFine)
            A += bg
            for i in range(2,NWorldFine[0]/2-1,2):
                for j in range(2,NWorldFine[1]/2-1,2):
                    A[i][j]= val
                    A[i+NWorldFine[0]/2-1][j]= val
                    #shaperemember
                    shapecounter += 1
                    S = np.append(S,[1,1,1])
                for k in range(NWorldFine[1]/2+1,NWorldFine[1]-2,2):
                    A[i][k]= val
                    A[i+NWorldFine[0]/2-1][k]= val
                    #shaperemember
                    shapecounter += 1
                    S = np.append(S,[1,1,1])
            
            S = np.reshape(S,(shapecounter,3))
            self.Matrix=A
            self.ShapeRemember = S
            self.ShapeRememberOriginal = S
            self.RandomMatrix = A
            return A
        
        
        
        np.random.seed(0)
        if self.probfactor > 0:
            valorbg = np.ones(self.probfactor)*bg                         #percentage
            valorbg[0] = val
        
        if self.probfactor < 0:
            valorbg = np.ones(-self.probfactor)*val                         #percentage
            valorbg[0] = bg
        
        LenList = [] 
        LenList.append(self.length)

        if LenSwitch is not None:
            assert(equidistant is None)
            LenList = LenSwitch #must be a list
    
        #channelspecial
        c = 0 # initial loop constant
        if ChannelVertical:
            assert(ChannelHorizontal is None)
            LenList = [NWorldFine[0]]
            c = 1
        if ChannelHorizontal:
            assert(ChannelVertical is None)
            LenList = [NWorldFine[1]]
            c = 1
        thickList = [] 
        thickList.append(thick)

        if thickSwitch is not None:
            assert(equidistant is None)
            thickList = thickSwitch #must be a list
    
    
        if ChannelVertical:
            starti = 0
            endi = NWorldFine[0]
        else:
            starti = 1
            endi = NWorldFine[0]-1
    
        if ChannelHorizontal:
            startj = 0
            endj = NWorldFine[1]
        else:
            startj = 1
            endj = NWorldFine[1]-1
    
        #Boundaryspace
        b = 0 #boundary initial
        if BoundarySpace:
            if space == 0:
                spacing = 1
            else:
                spacing = space
            if ChannelVertical is not True:
                starti = spacing
                endi = NWorldFine[0]-spacing
            if ChannelHorizontal is not True:
                startj = spacing
                endj = NWorldFine[1]-spacing
            b = spacing -1
        
        if self.TestExample:
            startj=22
            
        for i in range(starti,endi):
            #special for channel
            for j in range(startj,endj):
                # print "i" + str(i)
                # print "j" + str(j)
                #can we do something here?
                if A[i][j] == 0:
                    #will we do something here?
                    A[i][j] = random.sample(valorbg,1)[0]
                    if equidistant:
                        A[i][j] = 1 #yes sure
                    #if yes then
                    if A[i][j] == 1:
                        #len randomizing
                        Len = random.sample(LenList,1)[0]
                        thick = random.sample(thickList,1)[0]
                        #yes but first go back to zero
                        A[i][j] = 0
                        stop = 0 #initial for loop change
                        #build zuf
                        if ChannelVertical:
                            zuf = [4]
                        elif ChannelHorizontal:
                            zuf = [1]
                        
                        else:
                            zuf = []
                            zuf.extend([self.right*1,self.down*4,self.diagr1*2,self.diagr2*3,self.diagl1*5,self.diagl2*6])
                            for s in range(0,self.Shapes):
                                #IMMPORTANT
                                zuf.append(s+7)
                            zuf = list(filter(lambda x: x!=0 ,zuf))
                    
                        zuf1 = random.sample(zuf,1)[0] #chooses shape
                        '''
                        1 : right
                        2 : right diag1 
                        3 : right diag2
                        4 : down
                        5 : left diag1
                        6 : left diag2
                        7+:  new shapes
                        '''
                        
                        ########### investigate ##########
                        #does it fit into the grid
                        ShapeResults = []
                        
                        ShapeResults.append(self.InvestigateRight(A, i, j, Len, thick, b, c, Channel=ChannelHorizontal))
                        ShapeResults.append(self.InvestigateDiagr1(A, i, j, Len, thick, b, c))
                        ShapeResults.append(self.InvestigateDiagr2(A, i, j, Len, thick, b, c))
                        ShapeResults.append(self.InvestigateDown(A, i, j, Len, thick, b, c, Channel=ChannelVertical))
                        ShapeResults.append(self.InvestigateDiagl1(A, i, j, Len, thick, b, c))
                        ShapeResults.append(self.InvestigateDiagl2(A, i, j, Len, thick, b, c))
                        
                        for s in range(0,self.Shapes):
                            #rebuildMatrix
                            NewShapeMatrix = self.ShapeMatrixes[self.ShapeIndex[s]:self.ShapeIndex[s+1]]
                            NewShapeMatrix = np.reshape(NewShapeMatrix,(int(self.ShapeSizes[s][0]),int(self.ShapeSizes[s][1])))
                            ShapeCoords = self.ShapeCoords[self.CoordsIndex[s]:self.CoordsIndex[s+1]]
                            ShapeResults.append(self.InvestigateNewShapes(NewShapeMatrix, ShapeCoords, A, i, j, b, c))
                        
                    
                        for z in range(0,100):    #arbitrary
                            if ShapeResults[zuf1-1] == 0:
                                zuf1 = random.sample(zuf,1)[0]
                                stop = 1
                            else:
                                stop = 0
                                break
                                
                        if stop == 1:
                            continue
                        
                        #shape remember
                        shapecounter += 1
                        S = np.append(S,[zuf1,Len,thick])
                        ############################### keine if-abfragen zum crash mehr notwendig ###############
                        
                        if zuf1 == 1:
                            A, B = self.BuildRight(A, B, i, j, val, bg, Len, thick, space)
                        elif zuf1 == 2:
                            A, B = self.BuildDiagr1(A, B, i, j, val, bg, Len, thick, space)
                        elif zuf1 == 3:
                            A, B = self.BuildDiagr2(A, B, i, j, val, bg, Len, thick, space)
                        elif zuf1 == 4:
                            A, B = self.BuildDown(A, B, i, j, val, bg, Len, thick, space)
                        elif zuf1 == 5:
                            A, B = self.BuildDiagl1(A, B, i, j, val, bg, Len, thick, space)
                        elif zuf1 == 6:
                            A, B = self.BuildDiagl2(A, B, i, j, val, bg, Len, thick, space)
                        
                        for s in range(0,self.Shapes):
                            #rebuildMatrix
                            if zuf1 == 7+s:
                                NewShapeMatrix = self.ShapeMatrixes[self.ShapeIndex[s]:self.ShapeIndex[s+1]]
                                NewShapeMatrix = np.reshape(NewShapeMatrix,(int(self.ShapeSizes[s][0]),int(self.ShapeSizes[s][1])))
                                ShapeCoords = self.ShapeCoords[self.CoordsIndex[s]:self.CoordsIndex[s+1]]
                                A, B = self.BuildNewShapes(NewShapeMatrix, ShapeCoords, A, B, i, j, val, bg, space)
                        
        #search for all values
        valuecounter = 0
        for i in range(0,NWorldFine[0]):
            for j in range(0,NWorldFine[1]):
                if A[i][j] == 1:
                    valuecounter += 1
        
        self.valuecounter = valuecounter
                
        S = np.reshape(S,(shapecounter,3))
        #print S
        B += bg
        self.Matrix = B
        self.ShapeRemember = S
        self.ShapeRememberOriginal = S
        self.RandomMatrix = B
        return B
        
        
        ########################### investigation ##################################
    
    def InvestigateRight(self, A, i, j, Len, thick, b, c, inv = 1, Channel=None):
        if Channel:
            b1 = b
            b2 = 0
        else:
            b1 = b 
            b2 = b
            
        NWorldFine = self.NWorldFine
        result = 1
        if j+inv*(Len) < NWorldFine[1]-b2+c and j+inv*(Len) > -1+b2 and i+inv*(thick) < NWorldFine[0]-b1 and i+inv*(thick) > -1+b1:
            for k in range(0,int(inv*(Len)),inv):
                #rechts
                for l in range(0,int(inv*thick),inv):
                    if i+l < NWorldFine[0] and i+l > -1 and j+k < NWorldFine[1] and j+k > -1: 
                        if A[i+l][j+k] != 0:
                            result = 0
                    else:
                        result = 0
        else:
            result = 0
        return result
         
    def InvestigateDiagr1(self, A, i, j, Len, thick, b, c, inv = 1):
        NWorldFine = self.NWorldFine
        result = 1
        if j+inv*Len < NWorldFine[1]-b and j+inv*Len > -1+b and i+inv*(Len+1+thick-1) < NWorldFine[0]-b and i+inv*(Len+1+thick-1) > -1+b:
            for k in range(0,int(inv*Len),inv):
                #rechts diag1
                for l in range(0,int(inv*(thick+1)),inv):
                    if A[i+l+k][j+k] != 0:
                        result = 0
        else:
            result = 0
        return result
                
    
    def InvestigateDiagr2(self, A, i, j, Len, thick, b, c, inv = 1):
        NWorldFine = self.NWorldFine
        result = 1
        if j+inv*(Len+thick) < NWorldFine[1]-b and j+inv*(Len+thick) > -1+b and i+inv*(Len) < NWorldFine[0]-b and i+inv*(Len) > -1+b:
            for k in range(0,int(inv*Len),inv):
                #rechts diag2
                for l in range(0,int(inv*(thick+1)),inv):
                    if A[i+k][j+k+l] != 0:
                        result = 0
        else:
            result = 0
        return result 
      
    def InvestigateDown(self, A, i, j, Len, thick, b, c, inv = 1, Channel = None):
        if Channel:
            b1 = b
            b2 = 0
        else:
            b1 = b 
            b2 = b
        
        NWorldFine = self.NWorldFine
        result = 1
        if j+inv*(thick) < NWorldFine[1]-b1 and j+inv*(thick) > -1+b1 and i+inv*(Len)-c < NWorldFine[0]-b2 and i+inv*(Len) > -1 +b2:
            for k in range(0,int(inv*Len),inv):
                #down
                for l in range(0,int(inv*thick),inv):
                    if i+k < NWorldFine[0] and i+k > -1 and j+l < NWorldFine[1] and j+l > -1:
                        if A[i+k][j+l] != 0:
                            result= 0 
        
        else:
            result = 0
        return result
        
        
    def InvestigateDiagl1(self, A, i, j, Len, thick, b, c, inv = 1):
        NWorldFine = self.NWorldFine
        result = 1
        if j-inv*Len > -1+b and j-inv*Len < NWorldFine[1]-b and i+inv*(Len+thick) < NWorldFine[0]-b and i+inv*(Len+thick) >-1+b:
            for k in range(0,int(inv*Len),inv):
                #links diag1
                for l in range(0,int(inv*(thick+1)),inv):
                    if i+l+k < NWorldFine[0] and i+l+k > -1 and j-k < NWorldFine[1] and j-k > -1:
                        if A[i+l+k][j-k] != 0:
                            result = 0
        else:
            result = 0
        return result
                
        
    def InvestigateDiagl2(self, A, i, j, Len, thick, b, c, inv = 1):
        NWorldFine = self.NWorldFine
        result = 1
        if j-inv*(Len) > -1+b and j-inv*(Len) < NWorldFine[1] and i+inv*(Len) < NWorldFine[0]-b and i+inv*(Len) >-1+b and j+inv*(thick+1) < NWorldFine[1]-b and j+inv*(thick+1) > -1+b:
            for k in range(0,int(inv*Len),inv):
                #links diag2
                for l in range(0,int(inv*(thick+1)),inv):
                    if A[i+k][j-k+l] != 0:
                        result = 0
        
        else:
            result = 0
        return result
    
    def InvestigateNewShapes(self, ShapeBuildMatrix, ShapeCoords, A, i, j, b, c):
        NumberOfShapes = np.shape(ShapeBuildMatrix)[0]
        resulttotal = 0
        for shape in range(0,NumberOfShapes):
            #put them together uniquely with the following properties
            ShapeIndex = ShapeBuildMatrix[shape][0]                 #what shape
            ShapeLength = int(ShapeBuildMatrix[shape][1])                #what length
            ShapeThick = int(ShapeBuildMatrix[shape][2])                 #what thickness 
            ShapeInverser = int(ShapeBuildMatrix[shape][3])              #what direction
            px = ShapeCoords[shape*2]
            py = ShapeCoords[shape*2+1]
            
            #basic shape
            if ShapeIndex == 1:
                result = self.InvestigateRight(A, int(i+px), int(j+py), ShapeLength, ShapeThick, b, c, ShapeInverser)
            elif ShapeIndex == 2:
                result = self.InvestigateDiagr1(A, int(i+px), int(j+py), ShapeLength, ShapeThick, b, c, ShapeInverser)
            elif ShapeIndex == 3:
                result = self.InvestigateDiagr2(A,  int(i+px), int(j+py), ShapeLength, ShapeThick, b, c, ShapeInverser)
            elif ShapeIndex == 4:
                result = self.InvestigateDown(A, int(i+px), int(j+py), ShapeLength, ShapeThick, b, c, ShapeInverser)
            elif ShapeIndex == 5:
                result = self.InvestigateDiagl1(A,  int(i+px), int(j+py), ShapeLength, ShapeThick, b, c, ShapeInverser)
            elif ShapeIndex == 6:
                result = self.InvestigateDiagl2(A, int(i+px), int(j+py), ShapeLength, ShapeThick, b, c, ShapeInverser)
            
            resulttotal += result
        
        if np.sum(resulttotal) == NumberOfShapes:
            result = 1
        else:
            result = 0
            
        return result        

         
         ################################# Build #########################################
 
    def BuildRight(self, A, B, i, j, val, bg, Len, thick, space, inv = 1):
        NWorldFine = self.NWorldFine
        for k in range(0,int(inv*Len),inv):
            #rechts
            for l in range(0,int(inv*thick),inv):
                A[i+l][j+k] = 1
                B[i+l][j+k] = (val-bg)
        
            for s in range(inv,int(inv*(space+1)),inv):
                if i-s>-1 and i-s < NWorldFine[0]:
                    if A[i-s][j+k] != 1:
                        A[i-s][j+k] = bg
                if i+inv*(thick-1)+s < NWorldFine[0] and i+inv*(thick-1)+s > -1:
                    if A[i+int(inv*(thick-1))+s][j+k] != 1:
                        A[i+int(inv*(thick-1))+s][j+k] = bg
        
        for r in range(0,int(inv*(2*space+thick)),inv):
            for s in range(0,int(inv*space),inv):
                if i+r-inv*space < NWorldFine[0] and i+r-inv*space >-1 and j+inv*Len+s < NWorldFine[1] and j+inv*Len+s >-1:
                    if A[i+r-inv*space][j+int(inv*Len)+s] != 1:
                        A[i+r-inv*space][j+int(inv*Len)+s] = bg
                if i+r-inv*space < NWorldFine[0] and i+r-inv*space > -1 and j-inv-s > -1 and j-inv-s < NWorldFine[1]:
                    if A[i+r-inv*space][j-inv-s] != 1:
                        A[i+r-inv*space][j-inv-s] = bg
        return A, B
            
    def BuildDiagr1(self, A, B, i, j, val, bg, Len, thick, space, inv = 1):
        NWorldFine = self.NWorldFine
        for k in range(0,int(inv*Len),inv):
            #rechts diag1
            for l in range(0,int(inv*(thick+1)),inv):
                A[i+l+k][j+k] = 1
                B[i+l+k][j+k] = (val-bg)
        
            for s in range(inv,int(inv*(space+2)),inv):
                if i-s+k>-1 and i-s+k < NWorldFine[0]:
                    if A[i-s+k][j+k] != 1:
                        A[i-s+k][j+k] = bg
                if i+inv*thick+s+k < NWorldFine[0] and i+inv*thick+s+k >-1:
                    if A[i+inv*thick+s+k][j+k] != 1:
                        A[i+inv*thick+s+k][j+k] = bg    
                    
        for r in range(0,int(inv*(2*(space+1)+thick-1)),inv):
            for s in range(0,int(inv*space),inv):
                if i+inv*(Len-space-1)+r > -1 and i+inv*(Len-space-1)+r < NWorldFine[0] and j+inv*Len+s < NWorldFine[1] and j+inv*Len+s > -1:
                    if A[i+inv*(Len-space-1)+r][j+int(inv*Len)+s] != 1:
                        A[i+inv*(Len-space-1)+r][j+int(inv*Len)+s] = bg
    
        for r in range(0,int(inv*(2*space+thick+1)),inv):
            for s in range(0,int(inv*space),inv):
                if i+r-inv*space < NWorldFine[0] and i+r-inv*space > -1 and j-inv-s > -1 and j-inv-s < NWorldFine[1]:
                    if A[i+r-inv*space][j-inv-s] != 1:
                        A[i+r-inv*space][j-inv-s] = bg
    
        if i+inv*(Len+thick+space) < NWorldFine[0] and i+inv*(Len+thick+space) >-1:
            if A[i+inv*(Len+thick+space)][j+inv*(Len-1)] == bg:
                A[i+inv*(Len+thick+space)][j+inv*(Len-1)] = 0
        
        return A, B    
        
    def BuildDiagr2(self, A, B, i, j, val, bg, Len, thick, space, inv = 1):                        
        NWorldFine = self.NWorldFine
        for k in range(0,int(inv*Len),inv):
            #rechts diag2
            for l in range(0,int(inv*(thick+1)),inv):
                A[i+k][j+k+l] = 1
                B[i+k][j+k+l] = (val-bg)
        
            for s in range(inv,int(inv*(space+2)),inv):
                if j-s+k>-1 and j-s+k < NWorldFine[1] and i+k < NWorldFine[0] and i+k > -1:
                    if A[i+k][j-s+k] != 1:
                        A[i+k][j-s+k] = bg
                if j+inv*(thick)+s+k < NWorldFine[1] and j+inv*(thick)+s+k > -1:
                    if A[i+k][j+k+inv*thick+s] != 1:
                        A[i+k][j+k+inv*thick+s] = bg 
        
        for r in range(0,int(inv*(space)),inv):
            for s in range(0,inv*(2*space+1+thick),inv):
                if i-inv-r < NWorldFine[0] and i-inv-r >-1 and j-inv*space+s < NWorldFine[1] and j-inv*space+s > -1:
                    if A[i-inv-r][j-inv*space+s] != 1:
                        A[i-inv-r][j-inv*space+s] = bg
                    
        for r in range(0,int(inv*space),inv):
            for s in range(0,int(inv*(2*(space+1)+thick-1)),inv):
                if i+inv*Len+r < NWorldFine[0] and i+inv*Len+r > -1 and j+inv*(Len-space-1)+s < NWorldFine[1] and j+inv*(Len-space-1)+s > -1:
                    if A[i+inv*Len+r][j+inv*(Len-space-1)+s] != 1:
                        A[i+inv*Len+r][j+inv*(Len-space-1)+s] = bg

        if j+inv*(Len+thick+space) < NWorldFine[1]:
            if A[i+inv*(Len-1)][j+inv*(Len+thick+space)] == bg:
                A[i+inv*(Len-1)][j+inv*(Len+thick+space)] = 0
        
        return A, B 
                                          
    def BuildDown(self, A, B, i, j, val, bg, Len, thick, space, inv = 1):                         
        NWorldFine = self.NWorldFine
        for k in range(0,int(inv*Len),inv):
            #down
            for l in range(0,int(inv*thick),inv):
                A[i+k][j+l] = 1
                B[i+k][j+l] = (val-bg)
            
            for s in range(inv,int(inv*(space+1)),inv):
                if j-s>-1 and j-s < NWorldFine[1]:
                    if A[i+k][j-s] != 1:
                        A[i+k][j-s] = bg
                if j+inv*(thick-1)+s < NWorldFine[1] and j+inv*(thick-1)+s > -1:
                    if A[i+k][j+inv*(thick-1)+s] != 1:
                        A[i+k][j+inv*(thick-1)+s] = bg
        
        for r in range(0,int(inv*space),inv):
            for s in range(0,inv*(2*space+thick),inv):
                if i+inv*Len+r < NWorldFine[0] and i+inv*Len+r >-1 and j-inv*space+s < NWorldFine[1] and j-inv*space+s > -1:
                    if A[i+inv*Len+r][j-inv*space+s] != 1:
                        A[i+inv*Len+r][j-inv*space+s] = bg
        
        for r in range(0,int(inv*(space)),inv):
            for s in range(0,inv*(2*space+thick),inv):
                if i-inv-r < NWorldFine[0] and i-inv-r >-1 and j-inv*space+s < NWorldFine[1] and j-inv*space+s > -1:
                    if A[i-inv-r][j-inv*space+s] != 1:
                        A[i-inv-r][j-inv*space+s] = bg
        
        return A, B

    def BuildDiagl1(self, A, B, i, j, val, bg, Len, thick, space, inv = 1):
        NWorldFine = self.NWorldFine
        for k in range(0,int(inv*Len),inv):
            #links diag1
            for l in range(0,int(inv*(thick+1)),inv):
                A[i+l+k][j-k] = 1
                B[i+l+k][j-k] = (val-bg)
        
            for s in range(inv,int(inv*(space+2)),inv):
                if i-s+k>-1 and i-s+k < NWorldFine[0]:
                    if A[i-s+k][j-k] != 1:
                        A[i-s+k][j-k] = bg
                if i+inv*thick+s+k < NWorldFine[0] and i+inv*thick+s+k >-1:
                    if A[i+inv*thick+s+k][j-k] != 1:
                        A[i+inv*thick+s+k][j-k] = bg
                    
        for r in range(0,int(inv*(2*(space+1)+thick-1)),inv):
            for s in range(0,inv*space,inv):
                if i+inv*(Len-space-1)+r > -1 and i+inv*(Len-space-1)+r < NWorldFine[0] and j-inv*Len-s > -1 and j-inv*Len-s < NWorldFine[1]:
                    if A[i+inv*(Len-space-1)+r][j-inv*Len-s] != 1:
                        A[i+inv*(Len-space-1)+r][j-inv*Len-s] = bg
    
        for r in range(0,int(inv*(2*space+thick+1)),inv):
            for s in range(0,inv*space,inv):
                if i+r-inv*space < NWorldFine[0] and i+r-inv*space > -1 and j+inv+s < NWorldFine[1] and j+inv+s >-1:
                    if A[i+r-inv*space][j+inv+s] != 1:
                        A[i+r-inv*space][j+inv+s] = bg
    
        if i+inv*(Len+thick+space) < NWorldFine[0] and i+inv*(Len+thick+space) > -1:
            if A[i+inv*(Len+thick+space)][j+inv*(-Len+1)]==bg:
                A[i+inv*(Len+thick+space)][j+inv*(-Len+1)] = 0
        
        return A, B
    
    def BuildDiagl2(self, A, B, i, j, val, bg, Len, thick, space, inv = 1):                        
        NWorldFine = self.NWorldFine
        for k in range(0,int(inv*Len),inv):
            #links diag2
            for l in range(0,int(inv*(thick+1)),inv):
                A[i+k][j-k+l] = 1
                B[i+k][j-k+l] = (val-bg)
        
            for s in range(inv,int(inv*(space+2)),inv):
                if j-k+s+inv*thick < NWorldFine[1] and j-k+s+inv*thick >-1:
                    if A[i+k][j-k+s+inv*thick] != 1:
                        A[i+k][j-k+s+inv*thick] = bg
                if j-s-k > -1 and j-s-k < NWorldFine[1]:
                    if A[i+k][j-k-s] != 1:
                        A[i+k][j-k-s] = bg  
        
        for r in range(0,int(inv*space),inv):
            for s in range(0,int(inv*(2*(space+1)+thick-1)),inv):
                if i+inv*Len+r < NWorldFine[0] and i+inv*Len+r > -1 and j+inv*(1-Len-space)+s < NWorldFine[1] and j+inv*(1-Len-space)+s > -1:
                    if A[i+inv*Len+r][j+inv*(1-Len-space)+s] != 1:
                        A[i+inv*Len+r][j+inv*(1-Len-space)+s] = bg

        for r in range(0,int(inv*(space)),inv):
            for s in range(0,inv*(2*space+1+thick),inv):
                if i-inv-r < NWorldFine[0] and i-inv-r >-1 and j-inv*space+s < NWorldFine[1] and j-inv*space+s > -1:
                    if A[i-inv-r][j-inv*(space)+s] != 1:
                        A[i-inv-r][j-inv*(space)+s] = bg
                
        return A, B                          

    def BuildNewShapes(self, ShapeBuildMatrix, ShapeCoords, A, B, i, j, val, bg, space):
        NumberOfShapes = np.shape(ShapeBuildMatrix)[0]
        for shape in range(0,NumberOfShapes):
            #put them together uniquely with the following properties
            ShapeIndex = ShapeBuildMatrix[shape][0]                 #what shape
            ShapeLength = int(ShapeBuildMatrix[shape][1])                #what length
            ShapeThick = int(ShapeBuildMatrix[shape][2])               #what thick
            ShapeInverser = int(ShapeBuildMatrix[shape][3]) 
             
            px = ShapeCoords[shape*2]
            py = ShapeCoords[shape*2+1]
            
            #basic shape
            if ShapeIndex == 1:
                A, B = self.BuildRight(A, B, int(i+px), int(j+py), val, bg, ShapeLength, ShapeThick, space, ShapeInverser)
            elif ShapeIndex == 2:
                A, B = self.BuildDiagr1(A, B, int(i+px), int(j+py), val, bg, ShapeLength, ShapeThick, space, ShapeInverser)
            elif ShapeIndex == 3:
                A, B = self.BuildDiagr2(A, B, int(i+px), int(j+py), val, bg, ShapeLength, ShapeThick, space, ShapeInverser)
            elif ShapeIndex == 4:
                A, B = self.BuildDown(A, B, int(i+px), int(j+py), val, bg, ShapeLength, ShapeThick, space, ShapeInverser)
            elif ShapeIndex == 5:
                A, B = self.BuildDiagl1(A, B, int(i+px), int(j+py), val, bg, ShapeLength, ShapeThick, space, ShapeInverser)
            elif ShapeIndex == 6:
                A, B = self.BuildDiagl2(A, B, int(i+px), int(j+py), val, bg, ShapeLength, ShapeThick, space, ShapeInverser)
                
        return A,B        
    


    ################################# RANDOM ##############################################

    ################# Value Change ##############################
    
    
    def RandomValueChange(self, 
                    ratio=0.1, 
                    probfactor=1, 
                    randomvalue=None, 
                    negative=None, 
                    ShapeRestriction=True,
                    ShapeWave=None,
                    ChangeRight=1, 
                    ChangeDown=1, 
                    ChangeDiagr1=1, 
                    ChangeDiagr2=1, 
                    ChangeDiagl1=1, 
                    ChangeDiagl2=1,
                    Original = True,
                    NewShapeChange=True):
        #Changes Value randomly or certainly
        '''
        Ratio = amount of defect :    0.1 = 10 %  of the reference value 
        probfactor = defines the percentage of the possibility of the defect.   1 = 100% , 20 = 5%     maybe change this
        randomvalue = if true then a intervall of ratios is required
        negative = if true also negative defects are allowed
        '''
        #remember stuff
        assert(self.ShapeRemember is not None)
        assert(self.RandomMatrix is not None)
        NWorldFine = self.NWorldFine
        val = self.val
        
        C = self.RandomMatrix.copy()
        S = self.ShapeRemember.copy()
        
        if Original:
            C = self.Matrix.copy()
            S = self.ShapeRememberOriginal.copy()
        
        A = C.copy()
        #ratio
        ratioList = [ratio]
        if randomvalue is not None:
            ratioList = randomvalue
        
        #negative
        if negative:
            for i in range(0,np.size(ratioList)):
                ratioList.append(-ratioList[i])
        
        #probability
        if probfactor > 0:
            decision = np.zeros(probfactor)    
            decision[0] = 1
        
        if probfactor < 0:
            decision = np.ones(probfactor)    
            decision[0] = 0
        
        #for remember
        shapecounter = -1
        
        #NEwShapeChange
        ShapeChange = np.zeros(self.Shapes)
        if NewShapeChange == True:
            ShapeChange = np.ones(self.Shapes)
        else:
            for shape in range(0,int(self.Shapes)):
                if NewShapeChange[shape] == 0:
                    ShapeChange[shape] = 0 
                
        
        if ShapeRestriction:
            for i in range(0,NWorldFine[0]):
                for j in range(0,NWorldFine[1]):
                    if A[i][j]==1:
                        shapecounter += 1
                        #find the right shape
                        #regain the shape length and thicknes
                        zuf1 = int(S[shapecounter][0])
                        Len = int(S[shapecounter][1])
                        thick = int(S[shapecounter][2])
                        
                        ratiocur = random.sample(ratioList,1)[0]
                        decide = random.sample(decision,1)[0]
                        
                        if zuf1 == 1:
                            A, C = self.ValueChangeRight(A, C, i, j, ChangeRight, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick)
                        elif zuf1 == 2:
                            A, C = self.ValueChangeDiagr1(A, C, i, j, ChangeDiagr1, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick)
                        elif zuf1 == 3:
                            A, C = self.ValueChangeDiagr2(A, C, i, j, ChangeDiagr2, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick)
                        elif zuf1 == 4:
                            A, C = self.ValueChangeDown(A, C, i, j, ChangeDown, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick)
                        elif zuf1 == 5:
                            A, C = self.ValueChangeDiagl1(A, C, i, j, ChangeDiagl1, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick)
                        elif zuf1 == 6:
                            A, C = self.ValueChangeDiagl2(A, C, i, j, ChangeDiagl2, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick)
                        
                        for s in range(0,self.Shapes):
                            #rebuildMatrix
                            if zuf1 == 7+s:
                                NewShapeMatrix = self.ShapeMatrixes[self.ShapeIndex[s]:self.ShapeIndex[s+1]]
                                NewShapeMatrix = np.reshape(NewShapeMatrix,(int(self.ShapeSizes[s][0]),int(self.ShapeSizes[s][1])))
                                ShapeCoords = self.ShapeCoords[self.CoordsIndex[s]:self.CoordsIndex[s+1]]
                                A, C = self.ValueChangeNewShapes(NewShapeMatrix, ShapeCoords, A, C, i, j, ShapeChange[s], decide, decision, ShapeWave, ratioList, ratiocur, val)
                        
        else:
            for i in range(0,NWorldFine[0]):
                for j in range(0,NWorldFine[1]):
                    if A[i][j]==1:
                        if random.sample(decision,1)[0] == 1:
                            C[i][j] += random.sample(ratioList,1)[0] * val 
        
        self.RandomMatrix = C                
        return C    
    
    def ValueChangeRight(self, A, C, i, j, Change, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick, inv = 1):
        for k in range(0,inv*Len,inv):
            for l in range(0,inv*thick,inv):
                #forget about this shape
                A[i+l][j+k] = 0
            if Change:
                #rechts
                if decide == 1:
                    for l in range(0,inv*thick,inv):
                        #change it
                        if ShapeWave: 
                            if random.sample(decision,1)[0] == 1:
                                C[i+l][j+k] += random.sample(ratioList,1)[0] * val
                        else:
                            C[i+l][j+k] += ratiocur * val
        return A, C
    
    def ValueChangeDiagr1(self, A, C, i, j, Change, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick, inv = 1):
        for k in range(0,inv*Len,inv):
            #rechts diag1
            for l in range(0,inv*(thick+1),inv):
                A[i+l+k][j+k] = 0
            if Change:
                if decide == 1:
                    for l in range(0,inv*(thick+1),inv):
                        if ShapeWave: 
                            if random.sample(decision,1)[0] == 1:
                                C[i+l+k][j+k] += random.sample(ratioList,1)[0] * val
                        else:
                            C[i+l+k][j+k] += ratiocur * val
        return A, C
        
    def ValueChangeDiagr2(self, A, C, i, j, Change, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick, inv = 1):
        for k in range(0,inv*Len,inv):
            #rechts diag2
            for l in range(0,inv*(thick+1),inv):
                A[i+k][j+k+l] = 0
            if Change:
                if decide == 1:
                    for l in range(0,inv*(thick+1),inv):
                        if ShapeWave: 
                            if random.sample(decision,1)[0] == 1:
                                C[i+k][j+k+l] += random.sample(ratioList,1)[0] * val
                        else:
                            C[i+k][j+k+l] += ratiocur * val
        return A, C
        
    def ValueChangeDown(self, A, C, i, j, Change, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick, inv = 1):
        for k in range(0,inv*Len,inv):
            #down
            for l in range(0,inv*thick,inv):
                A[i+k][j+l] = 0
            if Change:
                if decide == 1:
                    for l in range(0,inv*thick,inv):
                        if ShapeWave: 
                            if random.sample(decision,1)[0] == 1:
                                C[i+k][j+l] += random.sample(ratioList,1)[0] * val
                        else:
                            C[i+k][j+l] += ratiocur * val
        
        return A, C
        
    def ValueChangeDiagl1(self, A, C, i, j, Change, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick, inv = 1):
        for k in range(0,inv*Len,inv):
            #links diag1
            for l in range(0,inv*(thick+1),inv):
                A[i+l+k][j-k] = 0
            if Change:
                if decide == 1:
                    for l in range(0,inv*(thick+1),inv):
                        if ShapeWave: 
                            if random.sample(decision,1)[0] == 1:
                                C[i+l+k][j-k] += random.sample(ratioList,1)[0] * val
                        else:
                            C[i+l+k][j-k] += ratiocur * val
        
        return A, C
        
    def ValueChangeDiagl2(self, A, C, i, j, Change, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick, inv = 1):    
        for k in range(0,inv*Len,inv):
            #links diag2
            for l in range(0,inv*(thick+1),inv):
                A[i+k][j-k+l] = 0
            
            if Change:
                if decide == 1:
                    for l in range(0,inv*(thick+1),inv):
                        if ShapeWave: 
                            if random.sample(decision,1)[0] == 1:
                                C[i+k][j-k+l] += random.sample(ratioList,1)[0] * val
                        else:
                            C[i+k][j-k+l] += ratiocur * val
        
        return A, C
    
    def ValueChangeNewShapes(self, ShapeBuildMatrix, ShapeCoords, A, C, i, j, ShapeChange, decide, decision, ShapeWave, ratioList, ratiocur, val):
        NumberOfShapes = np.shape(ShapeBuildMatrix)[0]
        for shape in range(0,NumberOfShapes):
            #put them together uniquely with the following properties
            ShapeIndex = ShapeBuildMatrix[shape][0]                 #what shape
            ShapeLength = int(ShapeBuildMatrix[shape][1])                #what length
            ShapeThick = int(ShapeBuildMatrix[shape][2])               #what thick
            ShapeInverser = int(ShapeBuildMatrix[shape][3]) 
             
            px = ShapeCoords[shape*2]
            py = ShapeCoords[shape*2+1]
            
            if ShapeIndex == 1:
                A, C = self.ValueChangeRight(A, C, int(i+px), int(j+py), ShapeChange, decide, decision, ShapeWave, ratioList, ratiocur, val, ShapeLength, ShapeThick, ShapeInverser)
            elif ShapeIndex == 2:
                A, C = self.ValueChangeDiagr1(A, C, int(i+px), int(j+py), ShapeChange, decide, decision, ShapeWave, ratioList, ratiocur, val, ShapeLength, ShapeThick, ShapeInverser)
            elif ShapeIndex == 3:
                A, C = self.ValueChangeDiagr2(A, C, int(i+px), int(j+py), ShapeChange, decide, decision, ShapeWave, ratioList, ratiocur, val, ShapeLength, ShapeThick, ShapeInverser)
            elif ShapeIndex == 4:
                A, C = self.ValueChangeDown(A, C, int(i+px), int(j+py), ShapeChange, decide, decision, ShapeWave, ratioList, ratiocur, val, ShapeLength, ShapeThick, ShapeInverser)
            elif ShapeIndex == 5:
                A, C = self.ValueChangeDiagl1(A, C, int(i+px), int(j+py), ShapeChange, decide, decision, ShapeWave, ratioList, ratiocur, val, ShapeLength, ShapeThick, ShapeInverser)
            elif ShapeIndex == 6:
                A, C = self.ValueChangeDiagl2(A, C, int(i+px), int(j+py), ShapeChange, decide, decision, ShapeWave, ratioList, ratiocur, val, ShapeLength, ShapeThick, ShapeInverser)
            
            
        return A,C    

    def SpecificValueChange(self, Number = None,
                    ratio=0.1, 
                    probfactor=1, 
                    randomvalue=None, 
                    negative=None, 
                    ShapeRestriction=True,
                    ShapeWave=None,
                    ChangeRight=1, 
                    ChangeDown=1, 
                    ChangeDiagr1=1, 
                    ChangeDiagr2=1, 
                    ChangeDiagl1=1, 
                    ChangeDiagl2=1,
                    Original = True,
                    NewShapeChange=True,
                    RandomSeed=None):
        #Changes Value randomly or certainly
        '''
        Ratio = amount of defect :    0.1 = 10 %  of the reference value 
        probfactor = defines the percentage of the possibility of the defect.   1 = 100% , 20 = 5%     maybe change this
        randomvalue = if true then a intervall of ratios is required
        negative = if true also negative defects are allowed
        '''

        if RandomSeed is not None:
            random.seed(RandomSeed)

        #remember stuff
        assert(self.ShapeRemember is not None)
        assert(self.RandomMatrix is not None)
        NWorldFine = self.NWorldFine
        val = self.val
        
        C = self.RandomMatrix.copy()
        S = self.ShapeRemember.copy()
        
        if Original:
            C = self.Matrix.copy()
            S = self.ShapeRememberOriginal.copy()
        
        A = C.copy()

        if Number is None:
            Number = [int(round(np.shape(S)[0]/2.,0))]
        
        #ratio
        ratioList = [ratio]
        if randomvalue is not None:
            ratioList = randomvalue
        
        #negative
        if negative:
            for i in range(0,np.size(ratioList)):
                ratioList.append(-ratioList[i])
        
        if probfactor > 0:
            decision = np.zeros(probfactor)    
            decision[0] = 1
        
        if probfactor < 0:
            decision = np.ones(probfactor)    
            decision[0] = 0
        
        #for remember
        shapecounter = -1
        
        #NEwShapeChange
        ShapeChange = np.zeros(self.Shapes)
        if NewShapeChange == True:
            ShapeChange = np.ones(self.Shapes)
        else:
            for shape in range(0,int(self.Shapes)):
                if NewShapeChange[shape] == 0:
                    ShapeChange[shape] = 0 
                
        
        if ShapeRestriction:
            for i in range(0,NWorldFine[0]):
                for j in range(0,NWorldFine[1]):
                    if A[i][j]==1:
                        shapecounter += 1
                        
                        
                        #find the right shape
                        #regain the shape length and thicknes
                        zuf1 = int(S[shapecounter][0])
                        Len = int(S[shapecounter][1])
                        thick = int(S[shapecounter][2])
                    
                        ratiocur = random.sample(ratioList,1)[0]
                        decide = random.sample(decision,1)[0]
                        
                        NumberList = filter(lambda x: x == shapecounter,Number)
                        if np.size(NumberList) == 1:
                            decide = 1
                        else:
                            decide = 0
                        
                        
                        NumberList = filter(lambda x: x == shapecounter,Number)
                        if np.size(NumberList) == 1:
                            move = 1
                        else:
                            move = 0
                        
                        
                        if zuf1 == 1:
                            A, C = self.ValueChangeRight(A, C, i, j, ChangeRight, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick)
                        elif zuf1 == 2:
                            A, C = self.ValueChangeDiagr1(A, C, i, j, ChangeDiagr1, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick)
                        elif zuf1 == 3:
                            A, C = self.ValueChangeDiagr2(A, C, i, j, ChangeDiagr2, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick)
                        elif zuf1 == 4:
                            A, C = self.ValueChangeDown(A, C, i, j, ChangeDown, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick)
                        elif zuf1 == 5:
                            A, C = self.ValueChangeDiagl1(A, C, i, j, ChangeDiagl1, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick)
                        elif zuf1 == 6:
                            A, C = self.ValueChangeDiagl2(A, C, i, j, ChangeDiagl2, decide, decision, ShapeWave, ratioList, ratiocur, val, Len, thick)
                    
                        for s in range(0,self.Shapes):
                            #rebuildMatrix
                            if zuf1 == 7+s:
                                NewShapeMatrix = self.ShapeMatrixes[self.ShapeIndex[s]:self.ShapeIndex[s+1]]
                                NewShapeMatrix = np.reshape(NewShapeMatrix,(int(self.ShapeSizes[s][0]),int(self.ShapeSizes[s][1])))
                                ShapeCoords = self.ShapeCoords[self.CoordsIndex[s]:self.CoordsIndex[s+1]]
                                A, C = self.ValueChangeNewShapes(NewShapeMatrix, ShapeCoords, A, C, i, j, ShapeChange[s], decide, decision, ShapeWave, ratioList, ratiocur, val)
                    
        self.RandomMatrix = C                
        return C    
            
    ##################################### Vanish ###################################    
                              
    def RandomVanish(self,  
                    probfactor=1,
                    PartlyVanish=None, 
                    ChangeRight=1, 
                    ChangeDown=1, 
                    ChangeDiagr1=1, 
                    ChangeDiagr2=1, 
                    ChangeDiagl1=1, 
                    ChangeDiagl2=1,
                    Original = True,
                    NewShapeChange=True,
                    RandomSeed=None):

        if RandomSeed is not None:
            random.seed(RandomSeed)
        #remember stuff
        assert(self.ShapeRememberOriginal is not None)
        assert(self.RandomMatrix is not None)
        NWorldFine = self.NWorldFine
        val = self.val
        bg = self.bg
        
        C = self.RandomMatrix.copy()
        S = self.ShapeRemember.copy()
        
        if Original:
            C = self.Matrix.copy()
            S = self.ShapeRememberOriginal.copy()
        
        A = C.copy()
        
        #probability
        if probfactor > 0:
            decision = np.ones(probfactor) * val    
            decision[0] = bg
        
        if probfactor < 0:
            decision = np.ones(probfactor) * bg   
            decision[0] = val
        
        #for remember
        shapecounter = -1
        
        #NEwShapeChange
        ShapeChange = np.zeros(self.Shapes)
        if NewShapeChange == True:
            ShapeChange = np.ones(self.Shapes)
        else:
            for shape in range(0,int(self.Shapes)):
                if NewShapeChange[shape] == 0:
                    ShapeChange[shape] = 0 
        
        
        for i in range(0,NWorldFine[0]):
            for j in range(0,NWorldFine[1]):
                if A[i][j]!=bg and A[i][j]!=0:
                    shapecounter += 1
                    #find the right shape
                    for s in range(shapecounter,np.shape(S)[0]):
                        if S[s][0] == 0:
                            shapecounter += 1
                        else:
                            break
                    
                    #regain the shape length and thicknes
                    zuf1 = int(S[shapecounter][0])
                    Len = int(S[shapecounter][1])
                    thick = int(S[shapecounter][2])
                    
                    vanish = random.sample(decision,1)[0]
                    
                    #initial diecounter
                    died = 0
                    
                    if zuf1 == 1:
                        A, C, died = self.VanishRight(A, C, i, j, Len, thick, ChangeRight, PartlyVanish, decision, vanish)
                    elif zuf1 == 2:
                        A, C, died = self.VanishDiagr1(A, C, i, j, Len, thick, ChangeDiagr1 , PartlyVanish, decision, vanish)
                    elif zuf1 == 3:
                        A, C, died = self.VanishDiagr2(A, C, i, j, Len, thick, ChangeDiagr2 , PartlyVanish, decision, vanish)
                    elif zuf1 == 4:
                        A, C, died = self.VanishDown(A, C, i, j, Len, thick, ChangeDown, PartlyVanish, decision, vanish)
                    elif zuf1 == 5:
                        A, C, died = self.VanishDiagl1(A, C, i, j, Len, thick, ChangeDiagl1, PartlyVanish, decision, vanish)
                    elif zuf1 == 6:
                        A, C, died = self.VanishDiagl2(A, C, i, j, Len, thick, ChangeDiagl2, PartlyVanish, decision, vanish)
                    
                    for s in range(0,self.Shapes):
                        #rebuildMatrix
                        if zuf1 == 7+s:
                            NewShapeMatrix = self.ShapeMatrixes[self.ShapeIndex[s]:self.ShapeIndex[s+1]]
                            NewShapeMatrix = np.reshape(NewShapeMatrix,(int(self.ShapeSizes[s][0]),int(self.ShapeSizes[s][1])))
                            ShapeCoords = self.ShapeCoords[self.CoordsIndex[s]:self.CoordsIndex[s+1]]
                            A, C, died = self.VanishNewShapes(NewShapeMatrix, ShapeCoords, A, C, i, j, ShapeChange[s], PartlyVanish, decision, vanish)
                    
                    if died == bg:
                        S[shapecounter][0] = 0
                        
        self.RandomMatrix = C           
        return C

    def VanishRight(self, A, C, i, j, Len, thick, Change, PartlyVanish, decision, vanish, inv = 1):
        died = 0
        for k in range(0,inv*Len,inv):
            for l in range(0,inv*thick,inv):
                #forget about this shape
                A[i+l][j+k] = 0
            if Change:
                #rechts
                for l in range(0,inv*thick,inv):
                    #change it
                    if PartlyVanish:
                        C[i+l][j+k] = random.sample(decision,1)[0]
                    else:
                        C[i+l][j+k] = vanish
                        died = vanish
        
        return A, C, died
    
    def VanishDiagr1(self, A, C, i, j, Len, thick, Change, PartlyVanish, decision, vanish, inv = 1):
        died = 0
        for k in range(0,inv*Len,inv):
            #rechts diag1
            for l in range(0,inv*(thick+1),inv):
                A[i+l+k][j+k] = 0
            if Change:
                for l in range(0,inv*(thick+1),inv):
                    if PartlyVanish:
                        C[i+l+k][j+k] = random.sample(decision,1)[0]
                    else:
                        C[i+l+k][j+k] = vanish
                        died = vanish
        return A, C, died
    def VanishDiagr2(self, A, C, i, j, Len, thick, Change, PartlyVanish, decision, vanish, inv = 1):
        died = 0
        for k in range(0,inv*Len,inv):
            #rechts diag2
            for l in range(0,inv*(thick+1),inv):
                A[i+k][j+k+l] = 0
            if Change:
                for l in range(0,inv*(thick+1),inv):
                    if PartlyVanish:
                        C[i+k][j+k+l] = random.sample(decision,1)[0]
                    else:
                        C[i+k][j+k+l] = vanish
                        died = vanish
        return A, C, died
        
    def VanishDown(self, A, C, i, j, Len, thick, Change, PartlyVanish, decision, vanish, inv = 1):
        died = 0
        for k in range(0,inv*Len,inv):
            #down
            for l in range(0,inv*thick,inv):
                A[i+k][j+l] = 0
            if Change:
                for l in range(0,inv*thick,inv):
                    if PartlyVanish:
                        C[i+k][j+l] = random.sample(decision,1)[0]
                    else:
                        C[i+k][j+l] = vanish
                        died = vanish
        return A, C, died
        
    def VanishDiagl1(self, A, C, i, j, Len, thick, Change, PartlyVanish, decision, vanish, inv = 1):
        died = 0
        for k in range(0,inv*Len,inv):
            #links diag1
            for l in range(0,inv*(thick+1),inv):
                A[i+l+k][j-k] = 0
               
            if Change:
                for l in range(0,inv*(thick+1),inv):
                    if PartlyVanish:
                        C[i+l+k][j-k] = random.sample(decision,1)[0]
                    else:
                        C[i+l+k][j-k] = vanish
                        died = vanish
        return A, C, died
        
    def VanishDiagl2(self, A, C, i, j, Len, thick, Change, PartlyVanish, decision, vanish, inv = 1):
        died = 0
        for k in range(0,inv*Len,inv):
            #links diag2
            for l in range(0,inv*(thick+1),inv):
                A[i+k][j-k+l] = 0
            
            if Change:
                for l in range(0,inv*(thick+1),inv):
                    if PartlyVanish:
                        C[i+k][j-k+l] = random.sample(decision,1)[0]
                    else:
                        C[i+k][j-k+l] = vanish
                        died = vanish
        return A, C, died
        
    def VanishNewShapes(self, ShapeBuildMatrix, ShapeCoords, A, C, i, j, ShapeChange, PartlyVanish, decision, vanish):
        died = 0
        NumberOfShapes = np.shape(ShapeBuildMatrix)[0]
        for shape in range(0,NumberOfShapes):
            #put them together uniquely with the following properties
            ShapeIndex = ShapeBuildMatrix[shape][0]                 #what shape
            ShapeLength = int(ShapeBuildMatrix[shape][1])                #what length
            ShapeThick = int(ShapeBuildMatrix[shape][2])               #what thick
            ShapeInverser = int(ShapeBuildMatrix[shape][3]) 
             
            px = ShapeCoords[shape*2]
            py = ShapeCoords[shape*2+1]
            
            if ShapeIndex == 1:
                A, C, died = self.VanishRight(A, C, int(i+px), int(j+py), ShapeLength, ShapeThick, ShapeChange, PartlyVanish, decision, vanish, ShapeInverser)
            elif ShapeIndex == 2:
                A, C, died = self.VanishDiagr1(A, C, int(i+px), int(j+py), ShapeLength, ShapeThick, ShapeChange, PartlyVanish, decision, vanish, ShapeInverser)
            elif ShapeIndex == 3:
                A, C, died = self.VanishDiagr2(A, C, int(i+px), int(j+py), ShapeLength, ShapeThick, ShapeChange, PartlyVanish, decision, vanish, ShapeInverser)
            elif ShapeIndex == 4:
                A, C, died = self.VanishDown(A, C, int(i+px), int(j+py), ShapeLength, ShapeThick, ShapeChange, PartlyVanish, decision, vanish, ShapeInverser)
            elif ShapeIndex == 5:
                A, C, died = self.VanishDiagl1(A, C, int(i+px), int(j+py), ShapeLength, ShapeThick, ShapeChange, PartlyVanish, decision, vanish, ShapeInverser)
            elif ShapeIndex == 6:
                A, C, died = self.VanishDiagl2(A, C, int(i+px), int(j+py), ShapeLength, ShapeThick, ShapeChange, PartlyVanish, decision, vanish, ShapeInverser)
            
            
        return A,C, died
        

    def SpecificVanish(self, Number = None, 
                    probfactor=1,
                    PartlyVanish=None, 
                    ChangeRight=1, 
                    ChangeDown=1, 
                    ChangeDiagr1=1, 
                    ChangeDiagr2=1, 
                    ChangeDiagl1=1, 
                    ChangeDiagl2=1,
                    Original = True,
                    NewShapeChange=True):

        #remember stuff
        assert(self.ShapeRememberOriginal is not None)
        assert(self.RandomMatrix is not None)
        NWorldFine = self.NWorldFine
        val = self.val
        bg = self.bg
            
        C = self.RandomMatrix.copy()
        S = self.ShapeRemember.copy()
        
        if Original:
            C = self.Matrix.copy()
            S = self.ShapeRememberOriginal.copy()
        
        A = C.copy()
            
        if Number is None:
            Number = [int(round(np.shape(S)[0]/2.,0))]
        
        
        #probability
        decision = np.ones(probfactor) * val    
        decision[0] = bg
        
        #for remember
        shapecounter = -1
        
        #NEwShapeChange
        ShapeChange = np.zeros(self.Shapes)
        if NewShapeChange == True:
            ShapeChange = np.ones(self.Shapes)
        else:
            for shape in range(0,int(self.Shapes)):
                if NewShapeChange[shape] == 0:
                    ShapeChange[shape] = 0 
        
        
        for i in range(0,NWorldFine[0]):
            for j in range(0,NWorldFine[1]):
                if A[i][j]!=bg and A[i][j]!=0:
                    shapecounter += 1
                    
                    NumberList = filter(lambda x: x == shapecounter,Number)
                    
                    if NumberList is not []:
                        #find the right shape
                        #regain the shape length and thicknes
                        for s in range(shapecounter,np.shape(S)[0]):
                            if S[s][0] == 0:
                                shapecounter += 1
                            else:
                                break
                            
                        zuf1 = int(S[shapecounter][0])
                        Len = int(S[shapecounter][1])
                        thick = int(S[shapecounter][2])
                    
                        vanish = random.sample(decision,1)[0]
                    
                        NumberList = filter(lambda x: x == shapecounter,Number)
                        if np.size(NumberList) == 1:
                            vanish = bg
                        else:
                            vanish = val
                        
                        #initial diecounter
                        died = 0
                    
                        if zuf1 == 1:
                            A, C, died = self.VanishRight(A, C, i, j, Len, thick, ChangeRight, PartlyVanish, decision, vanish)
                        elif zuf1 == 2:
                            A, C, died = self.VanishDiagr1(A, C, i, j, Len, thick, ChangeDiagr1 , PartlyVanish, decision, vanish)
                        elif zuf1 == 3:
                            A, C, died = self.VanishDiagr2(A, C, i, j, Len, thick, ChangeDiagr2 , PartlyVanish, decision, vanish)
                        elif zuf1 == 4:
                            A, C, died = self.VanishDown(A, C, i, j, Len, thick, ChangeDown, PartlyVanish, decision, vanish)
                        elif zuf1 == 5:
                            A, C, died = self.VanishDiagl1(A, C, i, j, Len, thick, ChangeDiagl1, PartlyVanish, decision, vanish)
                        elif zuf1 == 6:
                            A, C, died = self.VanishDiagl2(A, C, i, j, Len, thick, ChangeDiagl2, PartlyVanish, decision, vanish)
                    
                        for s in range(0,self.Shapes):
                            #rebuildMatrix
                            if zuf1 == 7+s:
                                NewShapeMatrix = self.ShapeMatrixes[self.ShapeIndex[s]:self.ShapeIndex[s+1]]
                                NewShapeMatrix = np.reshape(NewShapeMatrix,(int(self.ShapeSizes[s][0]),int(self.ShapeSizes[s][1])))
                                ShapeCoords = self.ShapeCoords[self.CoordsIndex[s]:self.CoordsIndex[s+1]]
                                A, C, died = self.VanishNewShapes(NewShapeMatrix, ShapeCoords, A, C, i, j, ShapeChange[s], PartlyVanish, decision, vanish)
                    
                        if died == bg:
                            S[shapecounter][0] = 0
                        
        self.RandomMatrix = C           
        return C

    
        ###################################### MOVE #############################################  
        
        
    def RandomMove(self,  
                    probfactor=1,
                    steps=1,
                    randomstep=None,
                    randomDirection=None, 
                    ChangeRight=1, 
                    ChangeDown=1, 
                    ChangeDiagr1=1, 
                    ChangeDiagr2=1, 
                    ChangeDiagl1=1, 
                    ChangeDiagl2=1,
                    Right=1,
                    BottomRight=0,
                    Bottom=0,
                    BottomLeft=0,
                    Left=0,
                    TopLeft=0,
                    Top=0,
                    TopRight=0,
                    Original = True,
                    NewShapeChange = True):

        #remember stuff
        assert(self.ShapeRememberOriginal is not None)
        assert(self.RandomMatrix is not None)
        
        NWorldFine = self.NWorldFine
        val = self.val
        bg = self.bg
        
        A = self.RandomMatrix.copy()
        S = self.ShapeRemember.copy()
        
        if Original:
            A = self.Matrix.copy()
            S = self.ShapeRememberOriginal.copy()
        
        C = np.zeros(NWorldFine)
        C += bg
        
        #probability
        if probfactor > 0:
            decision = np.zeros(probfactor)    
            decision[0] = 1
        
        if probfactor < 0:
            decision = np.ones(probfactor)    
            decision[0] = 0
        
        #steplist
        stepList = [steps]
        if randomstep is not None:
            stepList = randomstep
        
        MoveList = [Right*1,BottomRight*2,Bottom*3,BottomLeft*4,Left*5,TopLeft*6,Top*7,TopRight*8]
        MoveList = list(filter(lambda x: x!=0 ,MoveList))
        
        #for remember
        shapecounter = -1
        
        #initial boundaryfailcounter
        nomore = 0
        
        #NEwShapeChange
        ShapeChange = np.zeros(self.Shapes)
        if NewShapeChange == True:
            ShapeChange = np.ones(self.Shapes)
        else:
            for shape in range(0,int(self.Shapes)):
                if NewShapeChange[shape] == 0:
                    ShapeChange[shape] = 0 
        
        
        for i in range(0,NWorldFine[0]):
            for j in range(0,NWorldFine[1]):
                if A[i][j]!=bg and A[i][j]!=0:
                    shapecounter += 1
                    #find the right shape
                    #regain the shape length and thicknes
                    for s in range(shapecounter,np.shape(S)[0]):
                        if S[s][0] == 0:
                            shapecounter += 1
                        else:
                            break

                    zuf1 = int(S[shapecounter][0])
                    Len = int(S[shapecounter][1])
                    thick = int(S[shapecounter][2])
                    
                    move = random.sample(decision,1)[0]
                    step = random.sample(stepList,1)[0]
                    direction = random.sample(MoveList,1)[0]
                    
                    if direction == 1:
                        m1 = 0
                        m2 = step
                    elif direction == 2:
                        m1 = step
                        m2 = step
                    elif direction == 3:
                        m1 = step
                        m2 = 0
                    elif direction == 4:
                        m1 = step
                        m2 = -step
                    elif direction == 5:
                        m1 = 0
                        m2 = -step
                    elif direction == 6:
                        m1 = -step
                        m2 = -step
                    elif direction == 7:
                        m1 = -step
                        m2 = 0
                    elif direction == 8:
                        m1 = -step
                        m2 = step
                    
                    if randomDirection:
                        for i in range(0,np.size(stepList)):
                            stepList.append(-stepList[i])
                        stepList.append(0)
                        m1 = random.sample(stepList,1)[0]
                        m2 = random.sample(stepList,1)[0]
                    
                    if zuf1 == 1:
                        C = self.MoveRight(A, C, i, j, m1, m2, Len, thick, ChangeRight, move)
                        A, nomore = self.KillingRight(A, i, j, Len, thick)
                    elif zuf1 == 2:
                        C = self.MoveDiagr1(A, C, i, j, m1, m2, Len, thick, ChangeDiagr1, move)
                        A, nomore = self.KillingDiagr1(A, i, j, Len, thick)
                    elif zuf1 == 3:
                        C = self.MoveDiagr2(A, C, i, j, m1, m2, Len, thick, ChangeDiagr2, move)
                        A, nomore = self.KillingDiagr2(A, i, j, Len, thick)
                    elif zuf1 == 4:
                        C = self.MoveDown(A, C, i, j, m1, m2, Len, thick, ChangeDown, move)
                        A, nomore = self.KillingDown(A, i, j, Len, thick)
                    elif zuf1 == 5:
                        C = self.MoveDiagl1(A, C, i, j, m1, m2, Len, thick, ChangeDiagl1, move)
                        A, nomore = self.KillingDiagl1(A, i, j, Len, thick)
                    elif zuf1 == 6:
                        C = self.MoveDiagl2(A, C, i, j, m1, m2, Len, thick, ChangeDiagl2, move)
                        A, nomore = self.KillingDiagl2(A, i, j, Len, thick)
                    
                    for s in range(0,self.Shapes):
                        #rebuildMatrix
                        if zuf1 == 7+s:
                            NewShapeMatrix = self.ShapeMatrixes[self.ShapeIndex[s]:self.ShapeIndex[s+1]]
                            NewShapeMatrix = np.reshape(NewShapeMatrix,(int(self.ShapeSizes[s][0]),int(self.ShapeSizes[s][1])))
                            ShapeCoords = self.ShapeCoords[self.CoordsIndex[s]:self.CoordsIndex[s+1]]
                            A, C, died = self.MoveNewShapes(NewShapeMatrix, ShapeCoords, A, C, i, j, m1, m2, ShapeChange[s], move)
                                                            
        self.nomore = nomore
        self.RandomMatrix = C           
        return C
    
        ##### MOVE ###############      
    def MoveRight(self, A, C, i, j, m1, m2, Len, thick, Change, move, inv = 1):
        nomore = 0
        NWorldFine = self.NWorldFine
        for k in range(0,inv*Len,inv):
            if Change:
                if move ==1:
                    if i+inv*(thick)+m1 < NWorldFine[0] and i+inv*(thick)+m1 > -1 and j+inv*(Len)+m2 < NWorldFine[1] and j+inv*(Len)+m2 >-1:
                        #rechts
                        for l in range(0,inv*thick,inv):
                            #change it
                            C[i+l+m1][j+k+m2] = A[i+l][j+k]
                else:
                     for l in range(0,inv*thick,inv):
                         #change it
                         C[i+l][j+k] = A[i+l][j+k]
                    
        return C
        
    def MoveDiagr1(self, A, C, i, j, m1, m2, Len, thick, Change, move, inv = 1):    
        nomore = 0
        NWorldFine = self.NWorldFine
        for k in range(0,inv*Len,inv):
            #rechts diag1
            if Change:
                if move ==1:
                    if i+inv*(thick+Len-1)+m1 < NWorldFine[0] and i+inv*(thick+Len-1)+m1 > -1 and j+inv*(Len-1)+m2 < NWorldFine[1] and j+inv*(Len-1)+m2 >-1:
                        for l in range(0,inv*(thick+1),inv):
                            C[i+l+k+m1][j+k+m2] = A[i+l+k][j+k] 
                else:
                    for l in range(0,inv*(thick+1),inv):
                        C[i+l+k][j+k] = A[i+l+k][j+k]
        return C
        
    def MoveDiagr2(self, A, C, i, j, m1, m2, Len, thick, Change, move, inv = 1):
        nomore = 0
        NWorldFine = self.NWorldFine
        for k in range(0,inv*Len,inv):
            #rechts diag2
            if Change:
                if move ==1:
                    if i+inv*(Len-1)+m1 < NWorldFine[0] and i+inv*(Len-1)+m1 > -1 and j+inv*(thick+Len-1)+m2 < NWorldFine[1] and j+inv*(thick+Len-1)+m2 >-1:
                        for l in range(0,inv*(thick+1),inv):
                            C[i+k+m1][j+k+l+m2] = A[i+k][j+k+l]
                else:
                    for l in range(0,inv*(thick+1),inv):
                        C[i+k][j+k+l] = A[i+k][j+k+l]
        return C
        
    def MoveDown(self, A, C, i, j, m1, m2, Len, thick, Change, move, inv = 1):
        nomore = 0
        NWorldFine = self.NWorldFine
        for k in range(0,inv*Len,inv):
            #down
            if Change:
                if move ==1:
                    if i+inv*(Len-1)+m1 < NWorldFine[0] and i+inv*(Len-1)+m1 > -1 and j+inv*(thick-1)+m2 < NWorldFine[1] and j+inv*(thick-1)+m2 >-1:
                        for l in range(0,inv*thick,inv):
                            C[i+k+m1][j+l+m2] = A[i+k][j+l]
                else:
                    for l in range(0,inv*thick,inv):
                        C[i+k][j+l] = A[i+k][j+l]
        return C
        
    def MoveDiagl1(self, A, C, i, j, m1, m2, Len, thick, Change, move, inv = 1):
        nomore = 0
        NWorldFine = self.NWorldFine
        for k in range(0,inv*Len,inv):
            #links diag1
            if Change:
                if move ==1:
                    if i+inv*(Len-1+thick)+m1 < NWorldFine[0] and i+inv*(Len-1+thick)+m1 > -1 and j+inv*(1-Len)+m2 < NWorldFine[1] and j+inv*(1-Len)+m2 >-1:
                        for l in range(0,inv*(thick+1),inv):
                            C[i+l+k+m1][j-k+m2] = A[i+l+k][j-k]
                else:
                    for l in range(0,inv*(thick+1),inv):
                        C[i+l+k][j-k] = A[i+l+k][j-k]
        return C
        
    def MoveDiagl2(self, A, C, i, j, m1, m2, Len, thick, Change, move, inv = 1):
        nomore = 0
        NWorldFine = self.NWorldFine
        for k in range(0,inv*Len,inv):
            #links diag2
            if Change:
                if move ==1:
                    if i+inv*(Len-1)+m1 < NWorldFine[0] and i+inv*(Len-1)+m1 > -1 and j+inv*(thick+1-Len)+m2 < NWorldFine[1] and j+inv*(thick+1-Len)+m2 >-1:
                        for l in range(0,inv*(thick+1),inv):
                            C[i+k+m1][j-k+l+m2] = A[i+k][j-k+l]
                else:
                    for l in range(0,inv*(thick+1),inv):
                        C[i+k][j-k+l] = A[i+k][j-k+l]
        return C
    
        ########################## Killing ######################################
    def KillingRight(self, A, i, j, Len, thick, inv = 1):
        nomore = 0
        NWorldFine = self.NWorldFine
        for k in range(0,inv*Len,inv):
            for l in range(0,inv*thick,inv):
                if i+l < NWorldFine[0] and j+k < NWorldFine[1]:
                #forget about this shape
                    A[i+l][j+k] = 0
                else:
                    nomore = 1
        
        return A, nomore
        
    def KillingDiagr1(self, A, i, j, Len, thick, inv = 1):    
        nomore = 0
        NWorldFine = self.NWorldFine
        for k in range(0,inv*Len,inv):
            for l in range(0,inv*(thick+1),inv):
                if i+k < NWorldFine[0] and j+k < NWorldFine[1] and i+k >-1 and j+k >-1:
                    A[i+l+k][j+k] = 0
                else:
                    nomore = 1
        return A, nomore
        
    def KillingDiagr2(self, A, i, j, Len, thick, inv = 1):
        nomore = 0
        NWorldFine = self.NWorldFine
        for k in range(0,inv*Len,inv):
            for l in range(0,inv*(thick+1),inv):
                if i+k < NWorldFine[0] and j+k+l < NWorldFine[1]:
                    A[i+k][j+k+l] = 0
                else:
                    nomore = 1
        return A, nomore
        
    def KillingDown(self, A, i, j, Len, thick, inv = 1):
        nomore = 0
        NWorldFine = self.NWorldFine
        for k in range(0,inv*Len,inv):
            for l in range(0,inv*thick,inv):
                if i+k < NWorldFine[0] and j+l < NWorldFine[1]:
                    A[i+k][j+l] = 0
                else:
                    nomore = 1
        return A, nomore
        
    def KillingDiagl1(self, A, i, j, Len, thick, inv = 1):
        nomore = 0
        NWorldFine = self.NWorldFine
        for k in range(0,inv*Len,inv):
            for l in range(0,inv*(thick+1),inv):
                if i+k+l < NWorldFine[0] and j-k > -1:
                    A[i+l+k][j-k] = 0
                else:
                    nomore = 1
        return A, nomore
        
    def KillingDiagl2(self, A, i, j, Len, thick, inv = 1):
        nomore = 0
        NWorldFine = self.NWorldFine
        for k in range(0,inv*Len,inv):
            for l in range(0,inv*(thick+1),inv):
                if i+k < NWorldFine[0] and j-k+l < NWorldFine[1] and j-k+l > -1:
                    A[i+k][j-k+l] = 0
                else:
                    nomore = 1
        return A, nomore
     
    def MoveNewShapes(self, ShapeBuildMatrix, ShapeCoords, A, C, i, j, m1, m2, ShapeChange, move):
        nomore = 0
        NumberOfShapes = np.shape(ShapeBuildMatrix)[0]
        
        for shape in range(0,NumberOfShapes):
            #put them together uniquely with the following properties
            ShapeIndex = ShapeBuildMatrix[shape][0]                 #what shape
            ShapeLength = int(ShapeBuildMatrix[shape][1])                #what length
            ShapeThick = int(ShapeBuildMatrix[shape][2])               #what thick
            ShapeInverser = int(ShapeBuildMatrix[shape][3]) 
             
            px = ShapeCoords[shape*2]
            py = ShapeCoords[shape*2+1]
            
            if ShapeIndex == 1:
                C = self.MoveRight(A, C, int(i+px), int(j+py), m1, m2, ShapeLength, ShapeThick, ShapeChange, move, ShapeInverser)
            elif ShapeIndex == 2:
                C = self.MoveDiagr1(A, C, int(i+px), int(j+py), m1, m2, ShapeLength, ShapeThick, ShapeChange, move, ShapeInverser)
            elif ShapeIndex == 3:
                C = self.MoveDiagr2(A, C, int(i+px), int(j+py), m1, m2, ShapeLength, ShapeThick, ShapeChange, move, ShapeInverser)
            elif ShapeIndex == 4:
                C = self.MoveDown(A, C, int(i+px), int(j+py), m1, m2, ShapeLength, ShapeThick, ShapeChange, move, ShapeInverser)
            elif ShapeIndex == 5:
                C = self.MoveDiagl1(A, C, int(i+px), int(j+py), m1, m2, ShapeLength, ShapeThick, ShapeChange, move, ShapeInverser)
            elif ShapeIndex == 6:
                C = self.MoveDiagl2(A, C, int(i+px), int(j+py), m1, m2, ShapeLength, ShapeThick, ShapeChange, move, ShapeInverser)
            
        #killing
        for shape in range(0,NumberOfShapes):
            #put them together uniquely with the following properties
            ShapeIndex = ShapeBuildMatrix[shape][0]                 #what shape
            ShapeLength = int(ShapeBuildMatrix[shape][1])                #what length
            ShapeThick = int(ShapeBuildMatrix[shape][2])               #what thick
            ShapeInverser = int(ShapeBuildMatrix[shape][3]) 
         
            px = ShapeCoords[shape*2]
            py = ShapeCoords[shape*2+1]
            
            if ShapeIndex == 1:
                A, nomore = self.KillingRight(A, int(i+px), int(j+py), ShapeLength, ShapeThick, ShapeInverser)
            elif ShapeIndex == 2:
                A, nomore = self.KillingDiagr1(A, int(i+px), int(j+py), ShapeLength, ShapeThick, ShapeInverser)
            elif ShapeIndex == 3:
                A, nomore = self.KillingDiagr2(A, int(i+px), int(j+py),  ShapeLength, ShapeThick, ShapeInverser)
            elif ShapeIndex == 4:
                A, nomore = self.KillingDown(A, int(i+px), int(j+py),  ShapeLength, ShapeThick, ShapeInverser)
            elif ShapeIndex == 5:
                A, nomore = self.KillingDiagl1(A, int(i+px), int(j+py),  ShapeLength, ShapeThick, ShapeInverser)
            elif ShapeIndex == 6:
                A, nomore = self.KillingDiagl2(A, int(i+px), int(j+py),  ShapeLength, ShapeThick, ShapeInverser)
            
        return A,C, nomore
        
    def SpecificMove(self, Number = None,  
                    probfactor=1,
                    steps=1,
                    randomstep=None,
                    randomDirection=None, 
                    ChangeRight=1, 
                    ChangeDown=1, 
                    ChangeDiagr1=1, 
                    ChangeDiagr2=1, 
                    ChangeDiagl1=1, 
                    ChangeDiagl2=1,
                    Right=1,
                    BottomRight=0,
                    Bottom=0,
                    BottomLeft=0,
                    Left=0,
                    TopLeft=0,
                    Top=0,
                    TopRight=0,
                    AllDirections = False,
                    Original = True,
                    NewShapeChange = True,
                    RandomSeed = None):

        #remember stuff
        assert(self.ShapeRememberOriginal is not None)
        assert(self.RandomMatrix is not None)

        if RandomSeed is not None:
            random.seed(RandomSeed)

        NWorldFine = self.NWorldFine
        val = self.val
        bg = self.bg
        
        A = self.RandomMatrix.copy()
        S = self.ShapeRemember.copy()
        
        if Original:
            A = self.Matrix.copy()
            S = self.ShapeRememberOriginal.copy()
        
        C = np.zeros(NWorldFine)
        C += bg

        #probability
        decision = np.zeros(probfactor)     
        decision[0] = 1
        
        if Number is None:
            Number = [int(round(np.shape(S)[0]/2.,0))]
        
        
        #steplist
        stepList = [steps]
        if randomstep is not None:
            stepList = randomstep
        
        MoveList = [Right*1,BottomRight*2,Bottom*3,BottomLeft*4,Left*5,TopLeft*6,Top*7,TopRight*8]
        if AllDirections:
            MoveList = [r for r in range(1,9)]
        MoveList = list(filter(lambda x: x!=0 ,MoveList))
        
        #for remember
        shapecounter = -1
        
        #initial boundaryfailcounter
        nomore = 0
        
        #NEwShapeChange
        ShapeChange = np.zeros(self.Shapes)
        if NewShapeChange == True:
            ShapeChange = np.ones(self.Shapes)
        else:
            for shape in range(0,int(self.Shapes)):
                if NewShapeChange[shape] == 0:
                    ShapeChange[shape] = 0 
        
        
        for i in range(0,NWorldFine[0]):
            for j in range(0,NWorldFine[1]):
                if A[i][j]!=bg and A[i][j]!=0:
                    shapecounter += 1
                    #find the right shape
                    #regain the shape length and thicknes
                    for s in range(shapecounter,np.shape(S)[0]):
                        if S[s][0] == 0:
                            shapecounter += 1
                        else:
                            break

                    zuf1 = int(S[shapecounter][0])
                    Len = int(S[shapecounter][1])
                    thick = int(S[shapecounter][2])
            
                    move = random.sample(decision,1)[0]
                    step = random.sample(stepList,1)[0]
                    direction = random.sample(MoveList,1)[0]
                    
                    NumberList = filter(lambda x: x == shapecounter,Number)
                    if np.size(NumberList) == 1:
                        move = 1
                    else:
                        move = 0
            
            
                    if direction == 1:
                        m1 = 0
                        m2 = step
                    elif direction == 2:
                        m1 = step
                        m2 = step
                    elif direction == 3:
                        m1 = step
                        m2 = 0
                    elif direction == 4:
                        m1 = step
                        m2 = -step
                    elif direction == 5:
                        m1 = 0
                        m2 = -step
                    elif direction == 6:
                        m1 = -step
                        m2 = -step
                    elif direction == 7:
                        m1 = -step
                        m2 = 0
                    elif direction == 8:
                        m1 = -step
                        m2 = step
            
                    if randomDirection:
                        for i in range(0,np.size(stepList)):
                            stepList.append(-stepList[i])
                        stepList.append(0)
                        m1 = random.sample(stepList,1)[0]
                        m2 = random.sample(stepList,1)[0]
            
                    if zuf1 == 1:
                        C = self.MoveRight(A, C, i, j, m1, m2, Len, thick, ChangeRight, move)
                        A, nomore = self.KillingRight(A, i, j, Len, thick)
                    elif zuf1 == 2:
                        C = self.MoveDiagr1(A, C, i, j, m1, m2, Len, thick, ChangeDiagr1, move)
                        A, nomore = self.KillingDiagr1(A, i, j, Len, thick)
                    elif zuf1 == 3:
                        C = self.MoveDiagr2(A, C, i, j, m1, m2, Len, thick, ChangeDiagr2, move)
                        A, nomore = self.KillingDiagr2(A, i, j, Len, thick)
                    elif zuf1 == 4:
                        C = self.MoveDown(A, C, i, j, m1, m2, Len, thick, ChangeDown, move)
                        A, nomore = self.KillingDown(A, i, j, Len, thick)
                    elif zuf1 == 5:
                        C = self.MoveDiagl1(A, C, i, j, m1, m2, Len, thick, ChangeDiagl1, move)
                        A, nomore = self.KillingDiagl1(A, i, j, Len, thick)
                    elif zuf1 == 6:
                        C = self.MoveDiagl2(A, C, i, j, m1, m2, Len, thick, ChangeDiagl2, move)
                        A, nomore = self.KillingDiagl2(A, i, j, Len, thick)
            
                    for s in range(0,self.Shapes):
                        #rebuildMatrix
                        if zuf1 == 7+s:
                            NewShapeMatrix = self.ShapeMatrixes[self.ShapeIndex[s]:self.ShapeIndex[s+1]]
                            NewShapeMatrix = np.reshape(NewShapeMatrix,(int(self.ShapeSizes[s][0]),int(self.ShapeSizes[s][1])))
                            ShapeCoords = self.ShapeCoords[self.CoordsIndex[s]:self.CoordsIndex[s+1]]
                            A, C, died = self.MoveNewShapes(NewShapeMatrix, ShapeCoords, A, C, i, j, m1, m2, ShapeChange[s], move)
                           
                    
        self.nomore = nomore
        self.RandomMatrix = C           
        return C

    def ChannelVerticalRandomize(self, probfactor=10,
                         LU = 1,
                         RU = 1,
                         LO = 1,
                         RO = 1,
                         Original=True):
        assert(self.ChannelVertical)
        NWorldFine = self.NWorldFine
        
        bg = self.bg
        A = self.RandomMatrix.copy()
        
        if Original:
            A = self.Matrix.copy()
        
        B = self.Matrix.copy()
        
        if self.Channelsafer is None:
            CS = self.Matrix.copy()
        else:
            CS = self.Channelsafer
        
        O = self.Matrix.copy()
        
        if probfactor > 0:
            decision = np.zeros(probfactor)    
            decision[0] = 1
        
        if probfactor < 0:
            decision = np.ones(probfactor)    
            decision[0] = 0
        
        
        DirectionListori = [LU*1,RU*2,LO*3,RO*4]
        DirectionListori = list(filter(lambda x: x!=0 ,DirectionListori))
        
        for i in range(0,NWorldFine[0]):
            for j in range(0,NWorldFine[1]):
                if A[i][j]!=bg and A[i][j]!=0:
                    DirectionList = DirectionListori
                    
                    #get thick
                    thick = 0
                    for k in range(0,NWorldFine[1]):
                        if O[i][j+k]!=bg and O[i][j+k]!=0:
                            thick += 1
                            A[i][j+k] = 0
                        else:
                            break
                    
                    
                    #get length
                    spacelu = 0.
                    stop = 0
                    for k in range(1,NWorldFine[1]):
                        if stop == 1:
                            break
                        for l in range(0,thick):
                            if j-k > -1 and i-l > -1:
                                if CS[i-l][j-k] == bg or CS[i-l][j-k] == 0:
                                    spacelu += 1
                                else:
                                    stop = 1
                            else:
                                stop = 1
                                break
                    spacelu /= thick
                    
                    spacelo = 0.
                    stop = 0
                    for k in range(1,NWorldFine[1]):
                        if stop == 1:
                            break
                        for l in range(0,thick):
                            if j-k > -1 and i+l < NWorldFine[0]:
                                if CS[i+l][j-k] == bg or CS[i+l][j-k] == 0:
                                    spacelo += 1
                                else:
                                    stop = 1
                            else:
                                stop = 1
                                break
                    spacelo /= thick
                    
                    spacelreal = 0.
                    stop = 0
                    for k in range(1,NWorldFine[1]):
                        if j-k > -1:
                            if  O[i][j-k] == bg or O[i][j-k] == 0:
                                spacelreal += 1
                            else:
                                break
                        else:
                            break
                    
                    
                    spaceru = 0.
                    stop = 0
                    for k in range(1,NWorldFine[1]):
                        if stop == 1:
                            break
                        for l in range(0,thick):
                            if j+k+thick-1 < NWorldFine[1] and i-l > -1:
                                if CS[i-l][j+k+thick-1] == bg or CS[i-l][j+k+thick-1] == 0:
                                    spaceru += 1
                                else:
                                    stop = 1
                            else:
                                stop = 1
                                break
                    spaceru /= thick
                    
                    spacero = 0.
                    stop = 0
                    for k in range(1,NWorldFine[1]):
                        if stop == 1:
                            break
                        for l in range(0,thick):
                            if j+k+thick-1 < NWorldFine[1] and i+l < NWorldFine[0]:
                                if CS[i+l][j+k+thick-1] == bg or CS[i+l][j+k+thick-1] == 0:
                                    spacero += 1
                                else:
                                    stop = 1
                            else:
                                stop = 1
                                break
                    
                    spacero /= thick
                    
                    spacerreal = 0.
                    stop = 0
                    for k in range(1,NWorldFine[1]):
                        if j+k+thick-1 < NWorldFine[1]:
                            if O[i][j+k+thick-1] == bg or O[i][j+k+thick-1] == 0:
                                spacerreal += 1
                            else:
                                break
                        else:
                            break                    
                    
                    if spacelreal != spacelo:
                        DirectionList = list(filter(lambda x: x!=1 ,DirectionList))
                    
                    if spacelreal != spacelu:
                        DirectionList = list(filter(lambda x: x!=3 ,DirectionList))
                    
                    if spacerreal != spacero:
                        DirectionList = list(filter(lambda x: x!=2,DirectionList))
                    if spacerreal != spaceru:
                        DirectionList = list(filter(lambda x: x!=4,DirectionList))
                    
                    if spacelo ==0 or spacelu == 0:
                        DirectionList = list(filter(lambda x: x!=1 and x!=3 ,DirectionList))
                    if spacero ==0 or spaceru == 0:
                        DirectionList = list(filter(lambda x: x!=2 and x!=4 ,DirectionList))
                         
                    matu = 0
                    for k in range(1,NWorldFine[0]):
                        if i+k < NWorldFine[0]:
                            if CS[i+k][j]!=bg and CS[i+k][j]!=0:
                                matu += 1
                            else:
                                break
                        else:
                            break
                    
                    mato = 0
                    for k in range(1,NWorldFine[0]):
                        if i-k > -1:
                            if CS[i-k][j]!=bg and CS[i-k][j]!=0:
                                mato += 1
                            else:
                                break
                        else:
                            break
                    
                    if mato < spacelo or matu+1 < thick:
                        DirectionList = list(filter(lambda x: x!=3 ,DirectionList))
                    
                    if mato < spacero or matu+1 < thick:
                        DirectionList = list(filter(lambda x: x!=4 ,DirectionList))
                    
                    if matu < spacelu or mato+1 < thick:
                        DirectionList = list(filter(lambda x: x!=1 ,DirectionList))
                    
                    if matu < spaceru or mato+1 < thick:
                        DirectionList = list(filter(lambda x: x!=2 ,DirectionList))
                    
                    if np.size(DirectionList) != 0:
                        direction = random.sample(DirectionList,1)[0]
                    else:
                        continue
                    
                    if random.sample(decision,1)[0] == 0:
                        continue
                        
                    if direction == 1:
                        #LU
                        for k in range(1,int(spacelu+1)):
                            for l in range(0,thick):
                                B[i-l][j-k] = B[i+k][j+l]
                                CS[i+k][j+l] = 0
                                B[i+k][j+l] = bg
                                A[i+k][j+l] = 0
                                
                    elif direction == 2:
                        #Ru
                        for k in range(1,int(spaceru+1)):
                            for l in range(0,thick):
                                B[i-l][j+thick-1+k] = B[i+k][j+l]
                                CS[i+k][j+l] = 0
                                B[i+k][j+l] = bg
                                A[i+k][j+l] = 0
                        
                    elif direction == 3:
                        #LO
                        for k in range(1,int(spacelo+1)):
                            for l in range(0,thick):
                                B[i+l][j-k] = B[i-k][j+l]
                                CS[i-k][j+l] = 0
                                A[i-k][j+l] = 0
                                B[i-k][j+l] = bg
                        
                    elif direction == 4:
                        #RO
                        for k in range(1,int(spacero+1)):
                            for l in range(0,thick):
                                B[i+l][j+thick-1+k] = B[i-k][j+l]
                                CS[i-k][j+l] = 0
                                B[i-k][j+l] = bg
                                A[i-k][j+l] = 0
                    
                    
        self.Channelsafe = CS
        self.RandomMatrix = B                
        
        return B


    def ChannelHorizontalRandomize(self, probfactor=10,
                         LU = 1,
                         RU = 1,
                         LO = 1,
                         RO = 1,
                         Original=True):
        assert(self.ChannelHorizontal)
        NWorldFine = self.NWorldFine
        
        bg = self.bg
        A = self.RandomMatrix.copy()
        
        if Original:
            A = self.Matrix.copy()
        
        B = self.Matrix.copy()
        
        if self.Channelsafer is None:
            CS = self.Matrix.copy()
        else:
            CS = self.Channelsafer
        
        O = self.Matrix.copy()
        
        if probfactor > 0:
            decision = np.zeros(probfactor)    
            decision[0] = 1
        
        if probfactor < 0:
            decision = np.ones(probfactor)    
            decision[0] = 0
        
        
        DirectionListori = [LU*1,RU*2,LO*3,RO*4]
        DirectionListori = list(filter(lambda x: x!=0 ,DirectionListori))
        
        for i in range(0,NWorldFine[0]):
            for j in range(0,NWorldFine[1]):
                if A[i][j]!=bg and A[i][j]!=0:
                    DirectionList = DirectionListori
                    
                    #get thick
                    thick = 0
                    for k in range(0,NWorldFine[0]):
                        if O[i+k][j]!=bg and O[i+k][j]!=0:
                            thick += 1
                            A[i+k][j] = 0
                        else:
                            break
                    
                    
                    #get length
                    spacelu = 0.
                    stop = 0
                    for k in range(1,NWorldFine[0]):
                        if stop == 1:
                            break
                        for l in range(0,thick):
                            if i-k > -1 and j-l > -1:
                                if CS[i-k][j-l] == bg or CS[i-k][j-l] == 0:
                                    spacelu += 1
                                else:
                                    stop = 1
                            else:
                                stop = 1
                                break
                    spacelu /= thick
                    
                    spacelo = 0.
                    stop = 0
                    for k in range(1,NWorldFine[0]):
                        if stop == 1:
                            break
                        for l in range(0,thick):
                            if i-k > -1 and j+l < NWorldFine[1]:
                                if CS[i-k][j+l] == bg or CS[i-k][j+l] == 0:
                                    spacelo += 1
                                else:
                                    stop = 1
                            else:
                                stop = 1
                                break
                    spacelo /= thick
                    
                    spacelreal = 0.
                    stop = 0
                    for k in range(1,NWorldFine[0]):
                        if i-k > -1:
                            if  O[i-k][j] == bg or O[i-k][j] == 0:
                                spacelreal += 1
                            else:
                                break
                        else:
                            break
                    
                    
                    spaceru = 0.
                    stop = 0
                    for k in range(1,NWorldFine[0]):
                        if stop == 1:
                            break
                        for l in range(0,thick):
                            if i+k+thick-1 < NWorldFine[0] and j-l > -1:
                                if CS[i+k+thick-1][j-l] == bg or CS[i+k+thick-1][j-l] == 0:
                                    spaceru += 1
                                else:
                                    stop = 1
                            else:
                                stop = 1
                                break
                    spaceru /= thick
                    
                    spacero = 0.
                    stop = 0
                    for k in range(1,NWorldFine[0]):
                        if stop == 1:
                            break
                        for l in range(0,thick):
                            if i+k+thick-1 < NWorldFine[0] and j+l < NWorldFine[1]:
                                if CS[i+k+thick-1][j+l] == bg or CS[i+k+thick-1][j+l] == 0:
                                    spacero += 1
                                else:
                                    stop = 1
                            else:
                                stop = 1
                                break
                    
                    spacero /= thick
                    
                    spacerreal = 0.
                    stop = 0
                    for k in range(1,NWorldFine[0]):
                        if i+k+thick-1 < NWorldFine[0]:
                            if O[i+k+thick-1][j] == bg or O[i+k+thick-1][j] == 0:
                                spacerreal += 1
                            else:
                                break
                        else:
                            break                    
                    
                    if spacelreal != spacelo:
                        DirectionList = list(filter(lambda x: x!=1 ,DirectionList))
                    
                    if spacelreal != spacelu:
                        DirectionList = list(filter(lambda x: x!=3 ,DirectionList))
                    
                    #print DirectionList
                    
                    if spacerreal != spacero:
                        DirectionList = list(filter(lambda x: x!=2,DirectionList))
                    if spacerreal != spaceru:
                        DirectionList = list(filter(lambda x: x!=4,DirectionList))
                    
                    #print DirectionList
                    
                    if spacelo ==0 or spacelu == 0:
                        DirectionList = list(filter(lambda x: x!=1 and x!=3 ,DirectionList))
                    if spacero ==0 or spaceru == 0:
                        DirectionList = list(filter(lambda x: x!=2 and x!=4 ,DirectionList))
                         
                    #print DirectionList
                    matu = 0
                    for k in range(1,NWorldFine[1]):
                        if j+k < NWorldFine[1]:
                            if CS[i][j+k]!=bg and CS[i][j+k]!=0:
                                matu += 1
                            else:
                                break
                        else:
                            break
                    
                    mato = 0
                    for k in range(1,NWorldFine[1]):
                        if j-k > -1:
                            if CS[i][j-k]!=bg and CS[i][j-k]!=0:
                                mato += 1
                            else:
                                break
                        else:
                            break
                    
                    if mato < spacelo or matu < thick:
                        DirectionList = list(filter(lambda x: x!=3 ,DirectionList))
                    
                    if mato < spacero or matu < thick:
                        DirectionList = list(filter(lambda x: x!=4 ,DirectionList))
                    
                    if matu < spacelu or mato < thick:
                        DirectionList = list(filter(lambda x: x!=1 ,DirectionList))
                    
                    if matu < spaceru or mato < thick:
                        DirectionList = list(filter(lambda x: x!=2 ,DirectionList))
                    
                    if np.size(DirectionList) != 0:
                        direction = random.sample(DirectionList,1)[0]
                    else:
                        continue
                    
                    if random.sample(decision,1)[0] == 0:
                        continue
                        
                    if direction == 1:
                        #LU
                        for k in range(1,int(spacelu+1)):
                            for l in range(0,thick):
                                B[i-k][j-l] = B[i+l][j+k]
                                CS[i+l][j+k] = 0
                                B[i+l][j+k] = bg
                                A[i+l][j+k] = 0
                                
                    elif direction == 2:
                        #RU
                        for k in range(1,int(spaceru+1)):
                            for l in range(0,thick):
                                B[i+thick-1+k][j-l] = B[i+l][j+k]
                                CS[i+l][j+k] = 0
                                B[i+l][j+k] = bg
                                A[i+l][j+k] = 0
                        
                    elif direction == 3:
                        #LO
                        for k in range(1,int(spacelo+1)):
                            for l in range(0,thick):
                                B[i-k][j+l] = B[i+l][j-k]
                                CS[i+l][j-k] = 0
                                A[i+l][j-k] = 0
                                B[i+l][j-k] = bg
                        
                    elif direction == 4:
                        #RO
                        for k in range(1,int(spacero+1)):
                            for l in range(0,thick):
                                B[i+thick-1+k][j+l] = B[i+l][j-k]
                                CS[i+l][j-k] = 0
                                B[i+l][j-k] = bg
                                A[i+l][j-k] = 0
                    
                    
        self.Channelsafe = CS
        self.RandomMatrix = B                
        
        return B
        
    def ExtremeRandomizer(self, Vanish=True, ValueChange=None, Move=None, Original = True, Number=None):
        NWorldFine = self.NWorldFine
    
        bg = self.bg
        A = self.RandomMatrix.copy()
    
        if Original:
            A = self.Matrix.copy()
        
        if Number is None:
            Number = self.valuecounter[int(self.valuecounter/2.)]
        
        valuecounter = 0
        for i in range(0,NWorldFine[0]):
            for j in range(0,NWorldFine[1]):
                if A[i][j] == 1:
                    valuecounter += 1
                    NumberList = filter(lambda x: x == valuecounter,Number)
                    if np.size(NumberList) == 1:
                        if Vanish:
                            A[i][j] = bg
                        if ValueChange is not None:    
                            A[i][j] = ValueChange
                        if Move is not None:    
                            if Move[0]+i > -1 and Move[0]+i < NWorldFine[0] and Move[1]+j > -1 and Move[1]+j < NWorldFine[1]:
                                A[i+Move[0]][j+Move[1]] = A[i][j]  
                                A[i][j] = bg
        
        self.RandomMatrix = A
        return A
        