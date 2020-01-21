import sys
sys.path.append('tf_versions/2-1-0')
# sys.path.append('tf_versions/1-15-0')
import tensorflow as tf
tf_version = tf.__version__.replace('.', '_')
print(tf_version)
