from ngram import Ngram


def test():
    ng = Ngram()

    # Your n-gram model is trained with a text file
    ng.train('data/wiki-en-train.word')

    # You can save your trained model as text. Currently, we do not support loading trained model.
    ng.dump('trained_model')

    # You can evaluate your trained model with test file
    ng.test('data/wiki-en-test.word')

    # After your evaluation, you can retrieve the below results.
    print('log-likelihood\t={}'.format(ng.log_likelihood))
    print('entropy\t={}'.format(ng.entropy))
    print('perplexity\t={}'.format(ng.perplexity))
    print('coverage\t={}'.format(ng.coverage))


if __name__ == '__main__':
    test()
