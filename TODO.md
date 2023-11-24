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
- [ ] Make train and test set balanced by tissue using "gtex_tissue_path"
    - [ ] Challenge: You probably need to do stratified training, ensuring batches have samples from all tissues.
    - [ ] Opportunity: For your test-set, you should set aside all samples for a couple of tissues to test how the model generalizes to unseen tissues but also a proportion of samples from all samples to test for differences between seen and unseen tissues.
- [ ] Make PCA as baseline using the small gene file "gtex_gene_path"
    - [ ] Make PCA on this file
    - [ ] Make a model that using the PCA most representative data predicts isoforms in file "gtex_isoform_path"
- [ ] Make a VAE:
    - [ ] Train the latent space:
        - [ ] Start by using a subsample of "archs4_path" or "gtex_gene_path"
            - [ ] Make a latent space of 100 values to start, so we go from 18k values to 100
        - [ ] Using the big file for gene "archs4_path"
        - [ ] We can use a combination of "archs4_path" and "gtex_gene_path"
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

- Answers and questions:
    - Q: How should the isoform expressions be interpreted?
    - A: The higher the number the more there was of that RNA molecule within the cell
    - Q: What is the unit of the numbers?
    - A: Log2( TPM +1). TPM: Transcript Per Million. TPM is the normalization method that makes the numbers compariable across isoforms and samples.
    - Q: Furthermore, should the determination of isoform expressions be a classification problem or regression problem?
    - A: Regression problem. We want to predict the log2(TPM) values.

- [ ] Optional:
    - [ ] Explore possible gene visualizations:
        - [ ] [pyCirclize](https://github.com/moshi4/pyCirclize)
        - [ ] [pygenomeviz](https://pypi.org/project/pygenomeviz/)


