import trainer as tr
from SimpleCV import *


#classes = ['pear','orange','strawberry',]
classes = ['cocacola','heineken',]

def main():
        trainPaths = ['./post/'+c+'/train/' for c in classes ]
        testPaths =  ['./post/'+c+'/test/'   for c in classes ]

        trainer = tr.Trainer(classes,trainPaths)
        trainer.train()
        tree = trainer.classifiers[1]

        imgs = ImageSet()
        for p in testPaths:
                imgs += ImageSet(p)

        random.shuffle(imgs)

        print("Result test")
        trainer.test(testPaths)
        #trainer.visualizeResults(tree,imgs)


main()
