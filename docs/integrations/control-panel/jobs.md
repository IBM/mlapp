# Jobs

When you send a configuration, a new **job** is being created in the Control Panel. This **job** is responsible for running your configuration.

These jobs can be tracked and monitored in the **Jobs** page which is accessed via the navigation.

## 1. Jobs Table

In the jobs table you have the basic tools to monitor your jobs.

### Filtering
You can filter the jobs table by pipeline, status or free text search:
![job-screen-filtering](/integrations/control-panel/imgs/jobs-screen-filtering.png)

### Job details
Each row has different details about the job:
![job-screen-details](/integrations/control-panel/imgs/jobs-screen-details.png)

- **User**: which user has run the job.
- **Pipeline**: which pipeline is run in the configuration.
- **Asset Name**: which asset name is run in the configuration.
- **Asset Label**: which asset label is run in the configuration.
- **Status**: what status the job is at.
> Job Statuses:
>
> * _Running_ - currently running.
>
> * _In Queue_ - currently in queue and will be picked up when a resource is available.
>
> * _Complete_ - job completed.
>
> * _Failed_ - job failed, attached with the error message.
- **Started At**: time job started.
- **Ended At**: time job ended.

### Job Actions
There are some available actions you can do in the jobs page:
![job-screen-ations](/integrations/control-panel/imgs/jobs-screen-actions.png)

- **Refresh**: Refresh to see changes in the jobs' statuses.
- **Purge**: purge the queue from all jobs in it.

## 2. Job Information

When you click a row in the jobs table - a popup will show with more information on the job.

### Job Logs
In case you want to see the logs of the job, and specifically when you have errors you can view them in the **Logs** tab:
![job-logs](/integrations/control-panel/imgs/job-logs.png)

### Job Config
In case you want to see which configuration is attached to the job you can view it in the **Config** tab:
![job-config](/integrations/control-panel/imgs/job-config.png)