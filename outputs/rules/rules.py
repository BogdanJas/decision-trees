def findDecision(obj): #obj[0]: fixed acidity, obj[1]: volatile acidity, obj[2]: citric acid, obj[3]: residual sugar, obj[4]: chlorides, obj[5]: free sulfur dioxide, obj[6]: total sulfur dioxide, obj[7]: density, obj[8]: pH, obj[9]: sulphates, obj[10]: alcohol
   # {"feature": "alcohol", "instances": 1599, "metric_value": 0.9965, "depth": 1}
   if obj[10]<=11.488317414790277:
      # {"feature": "free sulfur dioxide", "instances": 1319, "metric_value": 0.9943, "depth": 2}
      if obj[5]>1.0:
         # {"feature": "chlorides", "instances": 1316, "metric_value": 0.994, "depth": 3}
         if obj[4]>0.038:
            # {"feature": "total sulfur dioxide", "instances": 1314, "metric_value": 0.9938, "depth": 4}
            if obj[6]<=112.9533211878411:
               # {"feature": "volatile acidity", "instances": 1239, "metric_value": 0.9985, "depth": 5}
               if obj[1]<=1.0736549872636059:
                  return 'Bad'
               elif obj[1]>1.0736549872636059:
                  return 'Bad'
               else:
                  return 'Bad'
            elif obj[6]>112.9533211878411:
               # {"feature": "residual sugar", "instances": 75, "metric_value": 0.3534, "depth": 5}
               if obj[3]<=9.516083208746958:
                  return 'Bad'
               elif obj[3]>9.516083208746958:
                  return 'Good'
               else:
                  return 'Bad'
            else:
               return 'Bad'
         elif obj[4]<=0.038:
            return 'Good'
         else:
            return 'Bad'
      elif obj[5]<=1.0:
         return 'Good'
      else:
         return 'Bad'
   elif obj[10]>11.488317414790277:
      # {"feature": "fixed acidity", "instances": 280, "metric_value": 0.4459, "depth": 2}
      if obj[0]>4.6:
         # {"feature": "sulphates", "instances": 279, "metric_value": 0.4352, "depth": 3}
         if obj[9]>0.37:
            # {"feature": "residual sugar", "instances": 278, "metric_value": 0.4241, "depth": 4}
            if obj[3]<=7.004917399898503:
               # {"feature": "volatile acidity", "instances": 272, "metric_value": 0.3923, "depth": 5}
               if obj[1]<=0.9722506001702707:
                  return 'Good'
               elif obj[1]>0.9722506001702707:
                  return 'Bad'
               else:
                  return 'Good'
            elif obj[3]>7.004917399898503:
               # {"feature": "chlorides", "instances": 6, "metric_value": 1.0, "depth": 5}
               if obj[4]<=0.051:
                  return 'Good'
               elif obj[4]>0.051:
                  return 'Bad'
               else:
                  return 'Good'
            else:
               return 'Good'
         elif obj[9]<=0.37:
            return 'Bad'
         else:
            return 'Good'
      elif obj[0]<=4.6:
         return 'Bad'
      else:
         return 'Good'
   else:
      return 'Good'
