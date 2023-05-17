from src.models.roberta import RobertaForPairwiseMatchingSubmission
from tianchi_args import *
from tianchi_init import *


def test_saved_model(saved_model_path):
    """
    The input_ids of query and doc are fake.
    Note:
        If you use bert-like model,
        *** don't forget add the index of '[CLS]' and '[SEP]' in the input_ids of query.
        *** don't forget add the index of '[SEP]' in the input_ids of doc.
    """
    query_input_ids = np.array([[101, 4508, 7942, 7000, 7350, 2586, 3296, 2225, 4275, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape(1, 128)
    doc_input_ids = np.array([[2408, 691, 7556, 705, 4385, 6573, 6862, 1355, 524, 11361, 120, 2608, 4448, 5687, 1788,
                               4508, 4841, 7000, 7350, 2364, 3296, 2225, 4275, 121, 119, 8132, 8181, 115, 8108, 4275,
                               120, 4665, 166, 8197, 5517, 7608, 5052, 5593, 4617, 3633, 1501, 3241, 3309, 5310, 1394,
                               6956, 5593, 4617, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [727, 1310, 4377, 4508, 4841, 7000, 796, 827, 3296, 2225, 5540, 1718, 9695, 8181, 115,
                               8114, 5108, 120, 4665, 5498, 5301, 5528, 4617, 1146, 1265, 1798, 4508, 4307, 5593, 4617,
                               2229, 6956, 3241, 3309, 3186, 5664, 2421, 2135, 3175, 3633, 1501, 102, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [928, 6449, 3710, 4508, 4384, 7000, 4275, 8114, 3710, 7185, 2128, 4508, 4384, 7000, 5790,
                               4508, 4841, 7000, 1079, 3302, 1366, 3302, 928, 2139, 7212, 1853, 3632, 6117, 5542, 4508,
                               7000, 1980, 6612, 3714, 4508, 3705, 3186, 5664, 2421, 3130, 1744, 772, 3709, 5790, 4275,
                               6820, 677, 3862, 4508, 3710, 3710, 7000, 5540, 1718, 5162, 7942, 7000, 3714, 2135, 3175,
                               102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0, 0]]).reshape(-1, 128)

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_path)
        tf.get_default_graph()
        query = sess.graph.get_tensor_by_name('query_ids:0')
        doc = sess.graph.get_tensor_by_name('doc_ids:0')
        score = sess.graph.get_tensor_by_name('score:0')
        score = sess.run(score, feed_dict={query: query_input_ids, doc: doc_input_ids})
        print("score: ", score)
        print("*** Please check if this score is correct.")
    print("*** The saved_model is available.")


def run_save_model_to_pb(ckpt_file, output_dir):
    query_ids = tf.placeholder(dtype=tf.int32, shape=[None, 128], name="query_ids")
    doc_ids = tf.placeholder(dtype=tf.int32, shape=[None, 128], name="doc_ids")
    config = BertConfig(seq_length=seq_length).from_json(config_file)
    model = RobertaForPairwiseMatchingSubmission(config)
    model.compile(training=False, query_input_ids=query_ids, doc_input_ids=doc_ids)
    model.load(ckpt_file)
    score = tf.identity(model.logits, name="score")
    model.save_to_pb(save_path=output_dir,
                     inputs={"query_ids": query_ids, "doc_ids": doc_ids},
                     outputs={"score": score})


if __name__ == '__main__':
    run_save_model_to_pb(checkpoint, wrapper_save_path)
    test_saved_model(wrapper_save_path)
