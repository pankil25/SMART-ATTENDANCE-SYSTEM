import datetime

from flask import Flask, render_template,request,session,redirect,url_for,Response
import sqlite3
import dataset,train,detection
import cv2
app = Flask(__name__)
app.secret_key = 'super secret key'
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/features',methods=['GET', 'POST'])
def features():
    return render_template('features.html')
@app.route('/aboutus/')
def contactus():
    return render_template('aboutus.html')


@app.route('/adminlogin/', methods = ["GET","post"])
def adminlogin():
    r =""
    msg=""
    if(request.method == "POST"):
        if(request.form["username"]!="" and request.form["password"]!=""):

            uname = request.form['username']
            pas = request.form['password']
            conn = sqlite3.connect('login.db')
            c = conn.cursor()
            c.execute("SELECT * FROM ad_auth WHERE username = '"+uname+"' and password = '"+pas+"'")
            r =  c.fetchall()
            for i in r:
                if(uname == i[0] and pas == i[1]):
                    session["logedin"]=True
                    session["username"]=uname
                    return redirect(url_for('admin'))
            msg = "Please Enter Valid Credentials"
        else:
            msg = "Please Enter Valid Credentials"
    return render_template("adminlogin.html",msg = msg)

@app.route('/admin',methods=['get','post'])
def admin():
    msg = ""
    msg2 = ""
    if (request.method == "POST"):
        if request.form.get("faculty"):
            if (request.form["id"] != "" and request.form["firstname"] != "" and request.form["lastname"] != "" and
                    request.form["mob"] != ""):
                ids = request.form['id']
                fname = request.form['firstname']
                lname = request.form['lastname']
                mob = request.form['mob']
                conn = sqlite3.connect('login.db')
                c = conn.cursor()
                c.execute("INSERT INTO faculty VALUES('" + ids + "','" + fname + "','" + lname + "','" + mob + "')")
                conn.commit()
                conn.close()
                msg = "Last Data Added Successfully"
            else:
                msg = "Enter All the Fields"
        elif request.form.get("student"):
            if (request.form["roll"] != "" and request.form["branch"] != "" and request.form["programme"] != "" and request.form["firstname"] != "" and request.form["lastname"] != "" and
                    request.form["mob"] != ""):
                roll = request.form['roll']
                branch = request.form['branch']
                programme = request.form['programme']
                fname = request.form['firstname']
                lname = request.form['lastname']
                mob = request.form['mob']


                imgs = dataset.take_pic(branch,roll,fname+'-'+lname)
                if(imgs>0):
                    conn = sqlite3.connect('login.db')
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO student (rollno, branch, programme, firstname, lastname, mobno) VALUES (?, ?, ?, ?, ?, ?)",
                        (roll, branch, programme, fname, lname, mob))

                    conn.commit()
                    conn.close()
                    msg = "Last Data Added Successfully"
                else:
                    msg = "Images should be taken more than one"
            else:
                msg = "Enter All the Fields"

        elif request.form.get("subject"):
            if (request.form["sname"] != "" and request.form["fid"] != "" and request.form["branch_name"] != "" and request.form["programme_name"] != ""):
                sname = request.form['sname']
                fid = request.form['fid']
                fid=fid.split('_')[0]
                branch_name = request.form['branch_name']
                programme_name = request.form['programme_name']
                conn = sqlite3.connect('login.db')
                c = conn.cursor()
                #c.execute("INSERT INTO subject VALUES('" + ids + "','" + sname + "','" + fid + "','" + branch_name + "')")
                c.execute(
                    "INSERT INTO subject (subname, facid, branch_name,programme) VALUES (?, ?, ?,?)",
                    (sname, fid, branch_name,programme_name))
                conn.commit()
                conn.close()
                msg = "Last Data Added Successfully"
            else:
                msg = "Enter All the Fields"



        elif request.form.get("branch"):
            if (request.form["branch"] != "" and request.form["intake"] != "" and request.form["programme"] != ""):
                branch_name = request.form['branch']
                intake = request.form['intake']
                programme_name = request.form['programme']
                conn = sqlite3.connect('login.db')
                c = conn.cursor()
                c.execute("INSERT INTO branch VALUES('" + branch_name + "','" + intake + "','" + programme_name + "')")
                conn.commit()
                conn.close()
                msg = "Last Data Added Successfully"
            else:
                msg = "Enter All the Fields"




        elif request.form.get("train"):
            temp = train.train("D:/Desktop/SAS/dataset",model_save_path="D:/Desktop/SAS/models/trained_model.clf",n_neighbors=3)
            if temp:
                msg2 = "Training Completed...!"
            else:
                msg2 = "Something is Wrong..."

    #request.form.get("view_student"):
    conn = sqlite3.connect('login.db')
    c = conn.cursor()
    c.execute("SELECT * FROM student")
    val = c.fetchall()
    c.execute("SELECT * FROM subject")
    sub = c.fetchall()
    c.execute("SELECT * FROM faculty")
    fac = c.fetchall()
    c.execute("SELECT * FROM branch")
    bra = c.fetchall()
    c.execute("SELECT DISTINCT branch FROM branch")
    uniquebranch=c.fetchall()
    c.execute("SELECT DISTINCT programme FROM branch")
    uniqueprogramme=c.fetchall()

    conn.commit()
    conn.close()

    return render_template('admin.html',msg=msg,val = val,sub=sub,fac=fac,bra=bra,uniqueprogramme=uniqueprogramme,uniquebranch=uniquebranch,msg2=msg2)

@app.route('/studentlogin/',methods = ["GET","post"])
def studentlogin():
    r = ""
    msg = ""
    if (request.method == "POST"):
        if (request.form["username"] != "" and request.form["password"] != ""):

            uname = request.form['username']
            pas = request.form['password']
            conn = sqlite3.connect('login.db')
            c = conn.cursor()
            c.execute("SELECT * FROM stu_auth WHERE username = '" + uname + "' and password = '" + pas + "'")
            r = c.fetchall()
            for i in r:
                    if (uname == i[0] and pas == i[1]):
                        session["logedin"] = True
                        session["username"] = uname
                        return redirect(url_for('student'))
            msg = "Please Enter Valid Credentials"
        else:
            msg = "Please Enter Valid Credentials"
    return render_template('studentlogin.html',msg = msg)


@app.route('/student',methods = ["GET","post"])
def student():
    conn = sqlite3.connect('login.db')
    c = conn.cursor()
    c.execute("SELECT DISTINCT subname FROM subject ")
    subj = c.fetchall()
    conn.close()
    msg = ""
    msg3 = ""
    if (request.method == "POST"):
        if request.form.get("view"):
            if (request.form["sub"] != "" and request.form["date"] != ""):
                uname = request.form["uname"]
                sub = request.form["sub"]
                temp = request.form["date"]
                date = temp[8:] + '-' + temp[5:7] + '-' + temp[0:4]
                conn = sqlite3.connect('login.db')
                conn2 = sqlite3.connect('attendance.db')
                c = conn.cursor()
                c2 = conn2.cursor()
                c.execute("SELECT roll_number FROM stu_auth WHERE username='"+uname+"'")
                id = c.fetchall()
                id = str(id[0][0])
                c.execute("SELECT branch FROM student WHERE rollno='"+id+"'")
                data = c.fetchall()
                b = data[0][0]
                roll = id
                table = str(b)+sub
                c2.execute(" SELECT count(name) FROM sqlite_master WHERE type='table' AND name='" + table + "'")
                if c2.fetchone()[0] != 1:
                    msg = "No Data Found"
                else:
                    c2.execute("SELECT count(roll) FROM '"+table+"' WHERE date='"+date+"' AND roll='"+roll+"'")
                    k = c2.fetchall()
                    if k[0][0] == 0:
                        msg = "You were marked Absent"
                    else:
                        msg = "You were marked Present"
            return render_template('student.html', sub=subj, msg=msg)

        if request.form.get("monthly"):
            print("In Monthly")
            if (request.form["sub"] != "" and request.form["month"] != ""):
                uname = request.form["uname2"]
                sub = request.form["sub"]
                temp = request.form["month"]

                my = temp[5:] + '-' + temp[0:4]
                print(my)
                conn = sqlite3.connect('login.db')
                conn2 = sqlite3.connect('attendance.db')
                c = conn.cursor()
                c2 = conn2.cursor()
                c.execute("SELECT roll_number FROM stu_auth WHERE username='" + uname + "'")
                id = c.fetchall()
                id = str(id[0][0])
                c.execute("SELECT branch FROM student WHERE rollno='" + id + "'")
                data = c.fetchall()
                b = data[0][0]
                roll = id
                table = str(b) + sub
                print(table)
                c2.execute(" SELECT count(name) FROM sqlite_master WHERE type='table' AND name='" + table + "'")
                if c2.fetchone()[0] != 1:
                    msg3 = "No Data Found"
                else:
                    c2.execute("SELECT * FROM '"+table+"' WHERE roll='"+roll+"' AND date LIKE ?", ('%'+my+'%',))
                    m = c2.fetchall()
                    if len(m)==0:
                        msg3 = "No Data Found"


            return render_template('student.html', sub=subj,m=m,msg3=msg3)


    return render_template('student.html',sub=subj)


@app.route('/facultylogin/',methods = ["GET","post"])
def facultylogin():
    r = ""
    msg = ""
    if (request.method == "POST"):
        if (request.form["username"] != "" and request.form["password"] != ""):

            uname = request.form['username']
            pas = request.form['password']
            conn = sqlite3.connect('login.db')
            c = conn.cursor()
            c.execute("SELECT * FROM fac_auth WHERE username = '" + uname + "' and password = '" + pas + "'")
            r = c.fetchall()
            for i in r:
                if (uname == i[0] and pas == i[1]):
                    session["logedin"] = True
                    session["username"] = uname
                    return redirect(url_for('faculty'))
            msg = "Please Enter Valid Credentials"
        else:
            msg = "Please Enter Valid Credentials"
    return render_template('facultylogin.html',msg=msg)
@app.route('/faculty',methods = ["GET","post"])
def faculty():
    det = []
    abs = []
    br = ''
    conn = sqlite3.connect('login.db')
    c = conn.cursor()
    uname = session["username"]
    c.execute("SELECT facid FROM fac_auth WHERE username='"+uname+"'")
    fid = c.fetchall()
    fid = str(fid[0][0])

    c.execute("SELECT DISTINCT branch_name FROM subject WHERE facid='"+fid+"'")
    branch = c.fetchall()

    c.execute("SELECT DISTINCT subname FROM subject WHERE facid='" + fid + "'")
    sub = c.fetchall()

    c.execute("SELECT DISTINCT programme FROM subject WHERE facid='" + fid + "'")
    programme = c.fetchall()

    conn.commit()
    conn.close()
    msg = ""
    msg3 = ""
    table2=""
    if (request.method == "POST"):

        if request.form.get("take"):
            if(request.form["sub"]!="" and request.form["branch"] != "" ):
                vid_cam = cv2.VideoCapture(0)
                b = request.form["branch"]

                s = request.form["sub"]

                return Response(detection.identify_faces(vid_cam,b,s),mimetype='multipart/x-mixed-replace; boundary=frame')
            else:
                msg = "Enter Required Feild"

        if request.form.get("view"):
            if (request.form["sub"] != "" and request.form["branch"] != ""  and request.form["date"] != ""):
                conn = sqlite3.connect('attendance.db')
                conn2 = sqlite3.connect('login.db')
                c = conn.cursor()
                c2 = conn2.cursor()
                b = request.form["branch"]
                sub = request.form["sub"]
                table = b + sub
                # Check if the table exists
                c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name=?", (table,))
                table_exists = c.fetchone()[0]

                # If table doesn't exist, create it
                if table_exists == 0:
                    create_table_query = f"""
                    CREATE TABLE "{table}" (
                        "name"	text,
                        "roll"	text,
                        "date"	text,
                        "time"	text,
                        PRIMARY KEY("roll","date")
                    )
                    """
                    c.execute(create_table_query)
                    conn.commit()  # Commit the table creation

                temp = request.form["date"]
                date = temp[8:]+'-'+temp[5:7]+'-'+temp[0:4]

                c.execute("SELECT * FROM '" + table + "'  where date= '" + date + "'")
                c2.execute("SELECT rollno,firstname,lastname FROM student WHERE branch = '" + b + "'  ")
                det2 = c2.fetchall()
                det = c.fetchall()
                flag = False
                for i in det2:
                    for j in det:
                        if(i[0]==j[1]):
                            flag = True

                    if flag==False:
                        abs.append((i[0], i[1], i[2],table))
                    flag = False

                c.close()
                c2.close()
                print(abs)
                conn.close()
                conn2.close()
            return render_template('view.html',det=det,abs=abs,branch=b,date=date)
        if request.form.get("mark"):
            roll = request.form["row0"]
            fn = request.form["row1"]
            ln = request.form["row2"]
            b = request.form["branch"]
            table = request.form["row3"]
            date=request.form["date"]
            name = fn+"-"+ln
            conn = sqlite3.connect('attendance.db')
            conn2 = sqlite3.connect('login.db')
            c = conn.cursor()
            c2 = conn2.cursor()
            now = datetime.datetime.now()
            #date = now.strftime("%d-%m-%Y")
            time = now.strftime("%I:%M:%S")
            c.execute(
                "INSERT INTO '"+table+"' (name, roll, date,time) VALUES (?, ?, ?,?)",
                (name, roll, date,time))
            conn.commit()
            # Check if the table exists
            c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name=?", (table,))
            table_exists = c.fetchone()[0]

            # If table doesn't exist, create it
            if table_exists == 0:
                create_table_query = f"""
                CREATE TABLE "{table}" (
                    "name"	text,
                    "roll"	text,
                    "date"	text,
                    "time"	text,
                    PRIMARY KEY("roll","date")
                )
                """
                c.execute(create_table_query)
                conn.commit()  # Commit the table creation
            c.execute("SELECT * FROM '" + table + "' WHERE date = '" + date + "' ")
            c2.execute("SELECT rollno,firstname,lastname FROM student WHERE branch = '" + b + "' ")
            det2 = c2.fetchall()
            det = c.fetchall()
            flag = False
            for i in det2:
                for j in det:
                    if (i[0] == j[1]):
                        flag = True

                if flag == False:
                    abs.append((i[0], i[1], i[2], table,date))
                flag = False
            c.close()
            c2.close()
            print(abs)
            conn.close()
            conn2.close()
            return render_template('view.html',det=det,abs = abs,date=date)
    return render_template('faculty.html',branch=branch,sub=sub,msg=msg,msg3=msg3)




@app.route('/logout')
def logout():
    session.clear()
    return render_template('index.html')




if __name__=="__main__":
    app.run(debug=True)