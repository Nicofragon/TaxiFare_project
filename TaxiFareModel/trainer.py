from TaxiFareModel.pipeline import TaxiFarePipeline
from TaxiFareModel.data_ import get_data, clean_data, holdout
from TaxiFareModel.utils import compute_rmse


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        '''returns a pipelined model'''

        tf_pipeline = TaxiFarePipeline()
        self.pipeline = tf_pipeline.create_pipeline()




    def run(self):
        """set and train the pipeline"""

        self.X_train, self.X_test, self.y_train, self.y_test = holdout(
            self.X, self.y)

        self.pipeline.fit(self.X_train, self.y_train)



    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""

        self.y_pred = self.pipeline.predict(self.X_test)
        rmse = compute_rmse(self.y_pred, self.y_test)
        return rmse


if __name__ == "__main__":
    # get data
    data = clean_data(get_data())
    # clean data
    # set X and y
    X = data[0]
    y = data[1]
    # hold out
    # train
    # evaluate
    test = Trainer(X,y)
    test.set_pipeline()
    test.run()
    result = test.evaluate()
    print(result)
