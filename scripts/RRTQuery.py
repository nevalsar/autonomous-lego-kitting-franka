import time
import pickle
import numpy as np

import RobotUtil as rt
from franka_robot import FrankaRobot 
from collision_boxes_publisher import CollisionBoxesPublisher
import Locobot
import rospy
import time
from frankapy import FrankaArm


def FindNearest(prevPoints,newPoint):
	D=np.array([np.linalg.norm(np.array(point)-np.array(newPoint)) 		for point in prevPoints])
	return D.argmin()

fr = FrankaRobot()

fa = FrankaArm()

deg_to_rad = np.pi/180.

#Initialize robot object
mybot=Locobot.Locobot()

pointsObs=[]
axesObs=[]


########  TODO: Fill in Box Parameters Here  ############################################
# envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.], [0.42, 0, .13]),[0.3, 0.1, 0.26])
# pointsObs.append(envpoints), axesObs.append(envaxes)

# position is: array([ 0.28772292, -0.07900434,  0.25534254])
# joints are: array([-0.33082418, -0.77368984,  0.06603997, -2.96390738,  0.07526653,
#         2.24546999,  0.5086481 ])
# position is: array([0.59773182, 0.02419393, 0.26472533])
# joints are: array([-0.30232313,  0.1765033 ,  0.33597939, -2.05138345,  0.03165035,
#         2.31673863,  0.79320665])


#########################################################################################

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.], [0.15, 0.46, 0.5]),[1.2, 0.01, 1.1])
pointsObs.append(envpoints), axesObs.append(envaxes)
envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.], [0.15, -0.46, 0.5]),[1.2, 0.01, 1.1])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.], [-0.41, 0, 0.5]),[0.01, 1, 1.11])
pointsObs.append(envpoints), axesObs.append(envaxes)
envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.], [0.75, 0, 0.5]),[0.01, 1, 1.1])
pointsObs.append(envpoints), axesObs.append(envaxes)
envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.], [0.2, 0, 1]),[1.2, 1, 0.01])
pointsObs.append(envpoints), axesObs.append(envaxes)
envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.], [0.2, 0, -0.05]),[ 1.2, 1, 0.01])
pointsObs.append(envpoints), axesObs.append(envaxes)


############# TODO: Define Start and Goal Joints #######################
qInit=np.array([ 0.15659579, -0.08702893,  0.32707317, -2.48130445, -0.0033423 ,
        2.42835672,  1.2255297 ])

qGoal=np.array([-0.52391983, -0.06294466, -0.0314536 , -2.43756123, -0.05292168,
        2.38287539,  0.38322698])

#TODO - Create RRT to find path to a goal configuration
rrtVertices=[]
rrtEdges=[]

rrtVertices.append(qInit)
rrtEdges.append(0)
thresh = 0.25
FoundSolution = False

while len(rrtVertices) < 5000 and not FoundSolution:
	print(len(rrtVertices))
	qRand = mybot.SampleRobotConfig()

	# Goal Bias
	if np.random.uniform(0,1) < 0.3:
		qRand = qGoal

	idNear = FindNearest(rrtVertices, qRand)
	qNear = rrtVertices[idNear]

	qRand = np.array(qRand)
	qNear = np.array(qNear)

	# Connect
	while np.linalg.norm(qRand-qNear) > thresh:
		qConnect = np.array(qNear) + (thresh * (np.array(qRand) - np.array(qNear)) / np.linalg.norm(qRand - qNear))
		if not mybot.DetectCollisionEdge(qNear, qConnect, pointsObs, axesObs):
			rrtVertices.append(qConnect)
			rrtEdges.append(idNear)
			qNear = qConnect
		else: 
			break

	qConnect = qRand

	if not mybot.DetectCollisionEdge(qNear,qConnect, pointsObs, axesObs):
		rrtVertices.append(qConnect)
		rrtEdges.append(idNear)

	idNear = FindNearest(rrtVertices, qGoal)

	print("current distance from obj: " + str(np.linalg.norm(np.asarray(qGoal) - np.asarray(rrtVertices[idNear]))))
	if np.linalg.norm(np.asarray(qGoal) - np.asarray(rrtVertices[idNear])) < 0.1:
		rrtVertices.append(qGoal)
		rrtEdges.append(idNear)
		FoundSolution = True
		print("Found solution")
		input("Press Enter to continue...")
		break

### if a solution was found

if FoundSolution:
	# Extract path
	plan = []
	c = -1  # Assume last added vertex is at goal
	plan.insert(0, rrtVertices[c])

	while True:
		c = rrtEdges[c]
		plan.insert(0, rrtVertices[c])
		if c == 0:
			break
	
	for i in range(150):
		anchorA = np.random.randint(0, len(plan) - 2)
		anchorB = np.random.randint(anchorA + 1, len(plan) - 1)
		
		shiftA = np.random.uniform(0,1)
		shiftB = np.random.uniform(0,1)

		#compute test vertices
		candidateA = (1-shiftA) * np.asarray(plan[anchorA]) + shiftA * np.asarray(plan[anchorA+1])
		#print(shiftA, shiftB, plan[anchorB], plan[anchorB+1])
		candidateB = (1-shiftB) * np.asarray(plan[anchorB]) + shiftB * np.asarray(plan[anchorB+1])

		#if no collision, shorten path
		if not mybot.DetectCollisionEdge(candidateA,candidateB,pointsObs,axesObs):
			while anchorB > anchorA:
				plan.pop(anchorB)
				anchorB = anchorB-1
			plan.insert(anchorA+1, candidateB)
			plan.insert(anchorA+1, candidateA)

	fa.reset_joints()
	for joint in plan:
		fa.goto_joints(joint)

else:
	print("No solution found")
