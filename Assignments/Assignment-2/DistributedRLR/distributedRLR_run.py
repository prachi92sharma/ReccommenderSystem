#!/usr/bin/env python

if __name__ == '__main__':
  
  import argparse
  import os
  import time

  script_dir = os.path.dirname(os.path.realpath(__file__))
  default_host_file = os.path.join(script_dir, 'hosts.txt')
  
  parser = argparse.ArgumentParser(description='Launches the JBosen DistributedRLR application using SSH.')
  parser.add_argument('--host_file', type=str, default=default_host_file, help='Path to the host file to use.')
  parser.add_argument('--num_local_worker_threads', type=int, default=4, help='Number of application worker threads per client.')
  parser.add_argument('--num_local_comm_channels', type=int, default=1, help='Number of network channels per client.')
  parser.add_argument('--output_dir', type=str, default=script_dir, help='Path to the output directory to use.')
  parser.add_argument('--vocSize', type=int, default=220000, help='Number of words in the vocabulary.')
  parser.add_argument('--learningRate', type=float, default=0.3, help='LearningRate')
  parser.add_argument('--regularization', type=float, default=0.003, help='Regularization')
  parser.add_argument('--epochs', type=int, default=3, help='Number of passes over the dataset to run.')
  parser.add_argument('--staleness', type=int, default=5, help='Number of clocks for each iteration.')
  parser.add_argument('--java_args', type=str, default='', help='Extra arguments to pass to Java.')
  parser.add_argument('--pem_file', type=str, default='',help='Location of AWS pem file')
  args = parser.parse_args()
  
  class_path = os.path.join(script_dir, 'build', 'libs', 'DistributedRLR.jar')
  main_class = 'DistributedRLR'

  with open(args.host_file, 'r') as f:
    host_ips = [line.split(':')[0] for line in f]

  def launch(client_id, ip):
    if args.pem_file:
        aws_args = "-i " + args.pem_file
    else:
        aws_args = " "
    cmd = 'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ' + aws_args + " " +  ip + ' '
    cmd += '"cd ' + os.getcwd() + '; '
    cmd += 'java ' + args.java_args + ' '
    cmd += '-cp ' + class_path + ' '
    cmd += main_class + ' '
    cmd += '-clientId %d ' % client_id
    cmd += '-hostFile %s ' % args.host_file
    cmd += '-numLocalWorkerThreads %d ' % args.num_local_worker_threads
    cmd += '-numLocalCommChannels %d ' % args.num_local_comm_channels
    cmd += '-outputDir %s ' % args.output_dir
    cmd += '-vocSize %d ' % args.vocSize
    cmd += '-learningRate %f ' % args.learningRate
    cmd += '-regularization %f ' % args.regularization
    cmd += '-epochs %d ' % args.epochs
    cmd += '-staleness %d ' % args.staleness
    cmd += '" &'
    print(cmd)
    os.system(cmd)

  print("Starting instances of DistributedRLR...")
  for client_id, ip in enumerate(host_ips):
    launch(client_id, ip)
