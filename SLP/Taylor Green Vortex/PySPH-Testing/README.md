# Taylor Green Vortex Simulation

## PySPH Testing

| Simulation Number 	|      Scheme      	|  PST  	|      Kernel     	| nx 	| perturb 	|  hdx 	| PST_Rh 	|
|:-----------------:	|:----------------:	|:-----:	|:---------------:	|:--:	|:-------:	|:----:	|:------:	|
|         00        	| $\delta^+$ - SPH 	|  TRUE 	| WendlandQuintic 	| 30 	|   0.2   	| 1.33 	|  0.05  	|
|         01        	| $\delta^+$ - SPH 	|  TRUE 	| WendlandQuintic 	| 50 	|   0.2   	| 1.33 	|  0.05  	|
|         02        	| $\delta^+$ - SPH 	|  TRUE 	| WendlandQuintic 	| 70 	|   0.2   	| 1.33 	|  0.05  	|
|         03        	| $\delta^+$ - SPH 	|  TRUE 	| WendlandQuintic 	| 90 	|   0.2   	| 1.33 	|  0.05  	|
|         04        	| $\delta^+$ - SPH 	|  TRUE 	| WendlandQuintic 	| 50 	|   0.2   	| 1.33 	|   0.1  	|
|         05        	| $\delta^+$ - SPH 	|  TRUE 	| WendlandQuintic 	| 50 	|   0.2   	| 1.33 	|   0.2  	|
|         06        	| $\delta^+$ - SPH 	|  TRUE 	| WendlandQuintic 	| 50 	|   0.2   	| 1.33 	|   0.5  	|
|         07        	| $\delta^+$ - SPH 	|  TRUE 	| WendlandQuintic 	| 50 	|   0.2   	| 1.33 	|    1   	|
|         08        	| $\delta^+$ - SPH 	|  TRUE 	| WendlandQuintic 	| 50 	|   0.2   	|   1  	|  0.05  	|
|         09        	| $\delta^+$ - SPH 	|  TRUE 	| WendlandQuintic 	| 50 	|   0.2   	|  1.5 	|  0.05  	|
|         10        	| $\delta^+$ - SPH 	|  TRUE 	| WendlandQuintic 	| 50 	|   0.2   	|   2  	|  0.05  	|
|         11        	| $\delta^+$ - SPH 	|  TRUE 	|  QuinticSpline  	| 50 	|   0.2   	|  1.5 	|  0.05  	|
|         12        	| $\delta^+$ - SPH 	|  TRUE 	|     Gaussian    	| 50 	|   0.2   	|  1.5 	|  0.05  	|
|         13        	| $\delta^+$ - SPH 	|  TRUE 	|   CubicSpline   	| 50 	|   0.2   	|  1.5 	|  0.05  	|
|         14        	| $\delta^+$ - SPH 	| FALSE 	| WendlandQuintic 	| 50 	|   0.2   	|  1.5 	|   ///  	|
|         15        	|  $\delta$ - SPH  	| FALSE 	|  QuinticSpline  	| 50 	|   0.2   	|   1  	|   ///  	|
|         16        	|       EDAC       	| FALSE 	|  QuinticSpline  	| 50 	|   0.2   	|   1  	|   ///  	|