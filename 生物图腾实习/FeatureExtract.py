import radiomics
import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import SimpleITK as sitk

def FeatureExtract(img,label):

    FirstOrderFeature = radiomics.firstorder.RadiomicsFirstOrder(img,label)
    FirstOrderFeature._initCalculation()
    A=[0]
    A[0]=FirstOrderFeature.getEnergyFeatureValue()  #1st
    A.append(FirstOrderFeature.getTotalEnergyFeatureValue()) #2nd
    A.append(FirstOrderFeature.getEntropyFeatureValue()) #3rd
    A.append(FirstOrderFeature.getMinimumFeatureValue()) #4th
    A.append(FirstOrderFeature.get10PercentileFeatureValue()) #5th
    A.append(FirstOrderFeature.get90PercentileFeatureValue()) #6th
    A.append(FirstOrderFeature.getMaximumFeatureValue()) #7th
    A.append(FirstOrderFeature.getMeanFeatureValue()) #8th
    A.append(FirstOrderFeature.getMedianFeatureValue()) #9th
    A.append(FirstOrderFeature.getInterquartileRangeFeatureValue()) #10th
    A.append(FirstOrderFeature.getRangeFeatureValue()) #11th
    A.append(FirstOrderFeature.getMeanAbsoluteDeviationFeatureValue()) #12th
    A.append(FirstOrderFeature.getRobustMeanAbsoluteDeviationFeatureValue()) #13th
    A.append(FirstOrderFeature.getRootMeanSquaredFeatureValue()) #14th
    A.append(FirstOrderFeature.getStandardDeviationFeatureValue()) #15th
    A.append(FirstOrderFeature.getSkewnessFeatureValue(axis=0)) #16th
    A.append(FirstOrderFeature.getKurtosisFeatureValue(axis=0)) #17th
    A.append(FirstOrderFeature.getVarianceFeatureValue()) #18th
    A.append(FirstOrderFeature.getUniformityFeatureValue()) #19th
    #print(A)

    ShapeFeature = radiomics.shape.RadiomicsShape(img,label)

    A.append(ShapeFeature.calculateDiameters())#20th


    A.append(ShapeFeature.getVolumeFeatureValue()) #21st
    A.append(ShapeFeature.getSurfaceAreaFeatureValue()) #22nd
    A.append(ShapeFeature.getSurfaceVolumeRatioFeatureValue()) #23rd
    A.append(ShapeFeature.getSphericityFeatureValue()) #24th
    A.append(ShapeFeature.getCompactness1FeatureValue()) #25th
    A.append(ShapeFeature.getCompactness2FeatureValue()) #26th
    A.append(ShapeFeature.getSphericalDisproportionFeatureValue()) #27th
    A.append(ShapeFeature.getMaximum3DDiameterFeatureValue()) #28th
    A.append(ShapeFeature.getMaximum2DDiameterSliceFeatureValue()) #29th
    A.append(ShapeFeature.getMaximum2DDiameterColumnFeatureValue()) #30th
    A.append(ShapeFeature.getMaximum2DDiameterRowFeatureValue()) #31st
    A.append(ShapeFeature.getMajorAxisFeatureValue()) #32nd
    A.append(ShapeFeature.getMinorAxisFeatureValue()) #33rd
    A.append(ShapeFeature.getLeastAxisFeatureValue()) #34th
    A.append(ShapeFeature.getElongationFeatureValue()) #35th
    A.append(ShapeFeature.getFlatnessFeatureValue()) #36th
    #print(B)

    GLCM = radiomics.glcm.RadiomicsGLCM(img,label)
    GLCM._initCalculation()

    A.append(GLCM.getAutocorrelationFeatureValue()) #37th
    A.append(GLCM.getJointAverageFeatureValue()) #38th
    A.append(GLCM.getClusterProminenceFeatureValue()) #39th
    A.append(GLCM.getClusterShadeFeatureValue()) #40th
    A.append(GLCM.getClusterTendencyFeatureValue()) #41st
    A.append(GLCM.getContrastFeatureValue()) #42nd
    A.append(GLCM.getCorrelationFeatureValue()) #43rd
    A.append(GLCM.getDifferenceAverageFeatureValue()) #44th
    A.append(GLCM.getDifferenceEntropyFeatureValue()) #45th
    A.append(GLCM.getDifferenceVarianceFeatureValue()) #46th

    #A.append(GLCM.getDissimilarityFeatureValue()) #47th
    A.append(0)

    A.append(GLCM.getJointEnergyFeatureValue()) #48th
    A.append(GLCM.getJointEntropyFeatureValue()) #49th

    #print(GLCM.getHomogeneity1FeatureValue()) #50th
    A.append(0)

    #print(GLCM.getHomogeneity2FeatureValue()) #51st
    A.append(0)

    A.append(GLCM.getImc1FeatureValue()) #52nd
    A.append(GLCM.getImc2FeatureValue()) #53rd
    A.append(GLCM.getIdmFeatureValue()) #54th
    A.append(GLCM.getIdmnFeatureValue()) #55th
    A.append(GLCM.getIdFeatureValue()) #56th
    A.append(GLCM.getIdnFeatureValue()) #57th
    A.append(GLCM.getInverseVarianceFeatureValue()) #58th
    A.append(GLCM.getMaximumProbabilityFeatureValue()) #59th
    A.append(GLCM.getSumAverageFeatureValue()) #60th

    #print(GLCM.getSumVarianceFeatureValue()) #61st
    A.append(0)

    A.append(GLCM.getSumEntropyFeatureValue()) #62nd
    A.append(GLCM.getSumSquaresFeatureValue()) #63rd
    #print(C)


    GLSZM = radiomics.glszm.RadiomicsGLSZM(img,label)
    GLSZM._initCalculation()

    A.append(GLSZM.getSmallAreaEmphasisFeatureValue()) #64th
    A.append(GLSZM.getLargeAreaEmphasisFeatureValue()) #65th
    A.append(GLSZM.getGrayLevelNonUniformityFeatureValue()) #66th
    A.append(GLSZM.getGrayLevelNonUniformityNormalizedFeatureValue()) #67th
    A.append(GLSZM.getSizeZoneNonUniformityFeatureValue()) #68th
    A.append(GLSZM.getSizeZoneNonUniformityNormalizedFeatureValue()) #69th
    A.append(GLSZM.getZonePercentageFeatureValue()) #70th
    A.append(GLSZM.getGrayLevelVarianceFeatureValue()) #71st
    A.append(GLSZM.getZoneVarianceFeatureValue()) #72nd
    A.append(GLSZM.getZoneEntropyFeatureValue()) #73rd
    A.append(GLSZM.getLowGrayLevelZoneEmphasisFeatureValue()) #74th
    A.append(GLSZM.getHighGrayLevelZoneEmphasisFeatureValue()) #75th
    A.append(GLSZM.getSmallAreaLowGrayLevelEmphasisFeatureValue()) #76th
    A.append(GLSZM.getSmallAreaHighGrayLevelEmphasisFeatureValue()) #77th
    A.append(GLSZM.getLargeAreaLowGrayLevelEmphasisFeatureValue()) #78th
    A.append(GLSZM.getLargeAreaHighGrayLevelEmphasisFeatureValue()) #79th
    #print(D)


    GLRLM = radiomics.glrlm.RadiomicsGLRLM(img,label)
    GLRLM._initCalculation()

    A.append(GLRLM.getShortRunEmphasisFeatureValue()) #80th
    A.append(GLRLM.getLongRunEmphasisFeatureValue()) #81st
    A.append(GLRLM.getGrayLevelNonUniformityFeatureValue()) #82nd
    A.append(GLRLM.getGrayLevelNonUniformityNormalizedFeatureValue()) #83rd00000000000000000000000000000000000000000000000000000000000000000000

    A.append(GLRLM.getRunLengthNonUniformityFeatureValue()) #84th
    A.append(GLRLM.getRunLengthNonUniformityNormalizedFeatureValue()) #85th
    A.append(GLRLM.getRunPercentageFeatureValue()) #86th
    A.append(GLRLM.getGrayLevelVarianceFeatureValue()) #87th
    A.append(GLRLM.getRunVarianceFeatureValue()) #88th
    A.append(GLRLM.getRunEntropyFeatureValue()) #89th
    A.append(GLRLM.getLowGrayLevelRunEmphasisFeatureValue()) #90th
    A.append(GLRLM.getHighGrayLevelRunEmphasisFeatureValue()) #91st
    A.append(GLRLM.getShortRunLowGrayLevelEmphasisFeatureValue()) #92nd
    A.append(GLRLM.getShortRunHighGrayLevelEmphasisFeatureValue()) #93rd
    A.append(GLRLM.getLongRunLowGrayLevelEmphasisFeatureValue()) #94th
    A.append(GLRLM.getLongRunHighGrayLevelEmphasisFeatureValue()) #95th
    #print(E)

    NGTDM = radiomics.ngtdm.RadiomicsNGTDM(img,label)
    NGTDM._initCalculation()

    A.append(NGTDM.getCoarsenessFeatureValue()) #96th
    A.append(NGTDM.getContrastFeatureValue()) #97th
    A.append(NGTDM.getBusynessFeatureValue()) #98th
    A.append(NGTDM.getComplexityFeatureValue()) #99th
    A.append(NGTDM.getStrengthFeatureValue()) #100th
    #print(F)

    GLDM = radiomics.gldm.RadiomicsGLDM(img,label)
    GLDM._initCalculation()

    A.append(GLDM.getSmallDependenceEmphasisFeatureValue()) #101st
    A.append(GLDM.getLargeDependenceEmphasisFeatureValue()) #102nd
    A.append(GLDM.getGrayLevelNonUniformityFeatureValue()) #103rd

    #print(GLDM.getGrayLevelNonUniformityNormalizedFeatureValue()) #104th
    A.append(0)

    A.append(GLDM.getDependenceNonUniformityFeatureValue()) #105th
    A.append(GLDM.getDependenceNonUniformityNormalizedFeatureValue()) #106sth
    A.append(GLDM.getGrayLevelVarianceFeatureValue()) #107th
    A.append(GLDM.getDependenceVarianceFeatureValue()) #108th
    A.append(GLDM.getDependenceEntropyFeatureValue()) #109th

    #print(GLDM.getDependencePercentageFeatureValue()) #110th
    A.append(0)

    A.append(GLDM.getLowGrayLevelEmphasisFeatureValue()) #111st
    A.append(GLDM.getHighGrayLevelEmphasisFeatureValue()) #112nd
    A.append(GLDM.getSmallDependenceLowGrayLevelEmphasisFeatureValue()) #113rd
    A.append(GLDM.getSmallDependenceHighGrayLevelEmphasisFeatureValue()) #114th
    A.append(GLDM.getLargeDependenceLowGrayLevelEmphasisFeatureValue()) #115th
    A.append(GLDM.getLargeDependenceHighGrayLevelEmphasisFeatureValue()) #116th
    #print(G)

    return(A)
