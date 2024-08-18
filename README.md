# ChainSum

Datasets and some related tools are available elsewhere due to the size limit; see: [Zenodo Record (anonymous)] (https://zenodo.org/records/13337169?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijc0NjliM2E2LTk5ZWUtNDM5YS05NDE4LThhYzJlYWU4ZTkxMSIsImRhdGEiOnt9LCJyYW5kb20iOiJkODY4Y2FkNmI4ODcwMzYzMWFmMDc3YmIzYjAxNDVhYSJ9.9mbUJdPrc-LNGCP48cl88r47mqNv_ikU2FMzwwfaMVZFnvxrZYgE_OkliGWq3Ay0TRmW1Dm8J7_57mio2cGkpA) (This link is static, so please make sure to select the latest version on the linked webpage if we update the file.)

First, download and unzip the file linked above. Then, place the `c2nl` and `data` directories under `/ChainSum/chainsum`.

To run the code, navigate to either `/ChainSum/chainsum/scripts/java` or `/ChainSum/chainsum/scripts/python` and execute the following command in the terminal:

```bash
bash transformer.sh [Parameter 1] [Parameter 2]
```

There are two parameters. 

(1) `[Parameter 1]` specifies the GPU ID(s). Set it to a specific number or several numbers separated by commas to enable the related GPU(s). On the other hand, no GPUs will be used if it is set to -1.

(2) `[Parameter 2]` can be any model name you choose to specify.

For example, the command can be: 
```bash
bash transformer.sh 0,1 code2jdoc
```

Once the code execution is complete, a directory `/ChainSum/chainsum/tmp` will be created, containing a series of files named after `[Parameter 2]`, including training logs and the generated summaries.

