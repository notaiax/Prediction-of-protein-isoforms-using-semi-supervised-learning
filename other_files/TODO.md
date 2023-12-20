## Paths:
- archs4_path = "/dtu-compute/datasets/iso_02456/archs4_gene_expression_norm_transposed.tsv.gz"
- gtex_gene_path = "/dtu-compute/datasets/iso_02456/gtex_gene_expression_norm_transposed.tsv.gz"
- gtex_isoform_path = "/dtu-compute/datasets/iso_02456/gtex_isoform_expression_norm_transposed.tsv.gz"
- gtex_anno_path = "/dtu-compute/datasets/iso_02456/gtex_gene_isoform_annoation.tsv.gz"
- gtex_tissue_path = "/dtu-compute/datasets/iso_02456/gtex_annot.tsv.gz"

## Deadlines:
- Poster presentation: The exam date is December 7th from 9 to 17
    -  The exam date is December 7th from 9 to 17. We divide the day into half hour slots and your group will later be given the possibility to register for a slot. A link to sign up for the poster session will appear here(https://docs.google.com/document/d/e/2PACX-1vQpXyHuor98w00pNnr9bN4qMO30TD-B8Tjrqvgj8vlNsr8cVzTJwrAyKB3SynriPbGRGFa_9GdFcEIm/pub) in due time. So having another exam on the same day should not be a problem.
- Final report deadline December 21st at 23:59



## Todo:
- [x] Explore data
    - [x] Explore datasets
    - [x] Try some vizualizations
- [ ] Make train and test set balanced by tissue using "gtex_tissue_path"
    - Challenge: You probably need to do stratified training, ensuring batches have samples from all tissues.
    - Opportunity: For your test-set, you should set aside all samples for a couple of tissues to test how the model generalizes to unseen tissues but also a proportion of samples from all samples to test for differences between seen and unseen tissues.
- [ ] Make PCA as baseline using the small gene file "gtex_gene_path"
    - [ ] Make PCA on this file
    - [ ] Make a model that using the PCA most representative data predicts isoforms in file "gtex_isoform_path"
- [ ] Make a VAE:
    - [X] Adapt DataLoader to use the load_data_chunk function, or use hd5 files if faster
    - [X] Train the latent space:
        - [X] Start by using a subsample of "archs4_path" or "gtex_gene_path"
            - [X] Make a latent space of 100 values to start, so we go from 18k values to 100
        - [X] Using the big file for gene "archs4_path"
        - [X] We can use a combination of "archs4_path" and "gtex_gene_path"
    - [ ] Improve VAE:
        - [ ] Standarize data to be between 0 and 1:
            - [ ] Search for max value in dataset or max level expression possible
            - [ ] Divide by it while loading data
        - We can see that our model generates genes with values that are wither 0 or 1, while a real gene have values that go from 0 to 14. Further steps include:
            - [X] Checking that model architecture is correct. We changed Bernoulli for LogNormal
            - [X] Improve model by augmenting latent space and epochs.
    - [X] Fix loss becoming NaN:
        - [X] Check there's no NaN values on input
        - [X] Add custom initialization to avoid NaN init values
            - [X] Try Kaiming
        - [X] Make learning rate smaller
            - [X] From 1e-3 to 1e-4
            - [ ] From 1e-3 to 1e-5
        - According to teacher probably the solution is making LR smaller, we can also plot the loss and see if it's too big if it makes big jumps. However, changing Bernoulli to LogNormal and LR=1e-4 solved the issue, we also removed Kaiming. 
        - Note: Adding small learning rate and kaiming solved the issues for most executions, but still getting NaN loss sometimes
        - Note 2: We removed kaiming and kept LR=1e-4, changing distribution to LogNormal solved the issue.
        - [ ] Add gradient clipping 
        - [ ] Check paper to see if I'm missing sth: https://arxiv.org/pdf/1406.5298.pdf
        - [ ] Review the Loss and ELBO Computation: Make sure that the computation of the loss function, especially the ELBO, is correct. Check for potential sources of numerical instability, such as logarithms of zero or negative values.
        - [ ] Batch Size: Sometimes, the choice of batch size can affect training stability. If your batch size is too small, it might lead to higher variance in the gradient updates. If it's too large, it might cause memory issues or affect the model's ability to generalize.
        - [ ] Check that data is correctly normalized
        - [ ] Regularization Techniques: Employ regularization techniques like dropout or weight decay to prevent overfitting and improve model stability.
        - [ ] Simplify model or train less data to see if problem persists, and then gradually scale up once we have a stable base model.
- [ ] Make a DNN:
    - [ ] Train the model that uses the latent space of the genes VAE to predict isoforms
        - [ ] Use the latent space of "gtex_gene_path" combined with "gtex_isoform_path" that contains the prediction results (or isoform values)
        - [ ] The model should be able to predict 150k float values, one for each isoform, using the 100 (or the number we choose) values from the latent space.
    - [ ] Improvement and issues:
        - [ ] To improve the model read the paper: [VAE Paper](https://arxiv.org/abs/1406.5298)
        - [ ] If you have issues handling the dataset, I have implemented it as a PyTorch dataset using the HDF5 format. You can randomly access the dataset without having to load it into memory. Read more about it on Slack.
- [ ] Make poster and prepare presentation:
    - [ ] A poster exam presentation, where the project groups document the results of their project in a poster and present to two or more teachers acting as examiners.
    - [ ] Plan for a 2-minute presentation per group member and 1-2 minutes for questions. The remainder of the time you can either present your poster to other students and guests or go visit other posters.
    - [ ] Remember that it is important for the overall impression that you divide the presentation and answering of the questions more or less equally between you.
    - [ ] The poster should be in A1 format. Remember to put both your names and student numbers under the title. Here and here are links to examples using the LaTeX template, and here is one in PowerPoint. You do not have to use that. The DTU library offers poster printing for a not too high price.
- [ ] Make report:
    - [ ] A report in which the project groups document their solution. The report should be a maximum of 6 pages plus references using this conference paper format: [Conference Paper Format](https://drive.google.com/file/d/0BxJRy96AHCJxaUEwOFhwUExmX00/view?resourcekey=0-RvwJqDVrZVijbkkifLWoYA)
    - [ ] The report should also contain a link to your project code GitHub repository. Among the files in the repository should be a Jupyter notebook that ideally should recreate the main results of your report. If some of your data is confidential, then use some shareable data instead.
    - [ ] For MSc students, please also include your poster in the submission.

## Answers and questions:
- Q: How should the isoform expressions be interpreted?
- A: The higher the number the more there was of that RNA molecule within the cell
- Q: What is the unit of the numbers?
- A: Log2( TPM +1). TPM: Transcript Per Million. TPM is the normalization method that makes the numbers compariable across isoforms and samples.
- Q: Furthermore, should the determination of isoform expressions be a classification problem or regression problem?
- A: Regression problem. We want to predict the log2(TPM) values.

## Optional:
- [ ] Explore possible gene visualizations:
    - [ ] [pyCirclize](https://github.com/moshi4/pyCirclize)
    - [ ] [pygenomeviz](https://pypi.org/project/pygenomeviz/)


