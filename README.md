# A-semi-supervised-framework-for-concept-based-hierarchical-document-clustering
https://doi.org/10.1007/s11280-023-01209-4



This is the Python implementation of the paper: <a href="https://doi.org/10.1007/s11280-023-01209-4">"A semi-supervised framework for concept-based hierarchical document clustering"</a> (Seyed Mojtaba Sadjadi, Hoda Mashayekhi & Hamid Hassanpour ) <br>
https://doi.org/10.1007/s11280-023-01209-4
<br>
<be>
<br>
<be>
<h4> Abstract </h4>
Text clustering is used in various applications of text analysis. In the clustering process, the employed document representation method has a significant impact on the results. Some popular document representation methods cannot effectively maintain the proximity information of the documents or suffer from low interpretability. Although the concept-based representation methods overcome these challenges to some extent, the existing semi-supervised document clustering methods rarely use this type of document representation. In this paper, we propose a concept-based semi-supervised framework for document clustering that uses both labeled and unlabeled data to yield a higher clustering quality. Concepts are composed of a set of semantically similar words. We propose the notion of semi-supervised concepts to benefit from document labels in extracting more relevant concepts. We also propose a new method of clustering documents based on the weights of such concepts. In the first and second steps of the proposed framework, the documents are represented based on the concepts extracted from the set of embedded words in the corpus. The proposed representation is interpretable and preserves the proximity information of documents. In the third step, the semi-supervised hierarchical clustering process utilizes unlabeled data to capture the overall structure of the clusters, and the supervision of a small number of labeled documents to adjust the cluster centroids. The use of concept vectors improves the process of merging clusters in the hierarchical clustering approach. The proposed framework is evaluated using the Reuters, 20-NewsGroups and WebKB text collections, and the results reveal the superiority of the proposed framework compared to several existing semi-supervised and unsupervised clustering approaches.
