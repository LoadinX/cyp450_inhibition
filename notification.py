import yagmail

def noreply(mailtolist = ['chidunxu@gmail.com'],mailuser = 'xmj@xiaoliming96.com',mailpass = 'XuMinJie19961127',mailhost = 'smtp.exmail.qq.com',subject = 'notification noreply mail',contents = ['this is a notification  mail , noreply'],title = 'noreply'):
    yag = yagmail.SMTP(user = mailuser,password = mailpass,host = mailhost)
    yag.send(mailtolist,subject,contents)
    print('{} mail sent'.format(title))

def JobDone(jobname):
    noreply(subject= f'[{jobname}] is done, plz check for next step',title = 'jobdone notification')



def RaiseError(jobname,error):
    standard_error_content = f'Hell No, we came across an error here:\n         {error}'
    noreply(subject= f'[{jobname}] interrupted', contents= standard_error_content,title = 'error notification')
    
if __name__ == '__main__':
    noreply()