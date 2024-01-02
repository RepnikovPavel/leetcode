#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>
using namespace std;


//  Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        std::unordered_map<int, int> h;
        std::vector<int> o_;
        auto N = nums.size();
        for(int i =0; i< N; i++){
            h.insert({nums[i],i});
        }
        for(int i=0;i<N;i++){
            auto ptr = h.find(target-nums[i]);
            if(ptr != h.end() && h[target-nums[i]]!=i){
                o_.push_back(i);
                o_.push_back(h[target-nums[i]]);
                break;
            }
        }
        return o_;
    }
    int romanToInt(string s) {
        int N = s.size();
        auto h = std::unordered_map<int,int>{
            {'I',1},
            {'V',5},
            {'X',10},
            {'L',50},
            {'C',100},
            {'D',500},
            {'M',1000}
        };
        auto mulh = std::unordered_map<std::string, int>{
            {"IV", 4},
            {"IX", 9},
            {"XL", 40},
            {"XC", 90},
            {"CD",400},
            {"CM",900}
        };

        int sum_ = 0;
        int i = N-1;
        while(i>-1){
            if(i==0){
                sum_ += h[s[i]];
                break;
            }
            auto sub_ = s.substr(i-1,2);
            if (mulh.find(sub_) != mulh.end())
            {
                sum_ += mulh[sub_];
                i -= 2;
            }
            else {
                // sum_ += h[sub_[0]];
                sum_ += h[sub_[1]];
                i -=1;
            }
        }            
        return sum_;
    }
    bool hasCycle(ListNode *head) {
        if(head == nullptr){
            return false;
        }
        auto h = std::unordered_map<ListNode*, int>();
        auto left = head/* condition */->next;
        ListNode* cv = head;
        int pos = 0;
        while(left != nullptr){
            if(h.find(cv) != h.end()){
                return true;
            }
            else{
                h.insert({cv, pos});
            }
            cv = left->next;
            left = left->next;
            pos += 1;
        }
        return false;
    }

    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB){
        if (headA==nullptr && headB != nullptr)
        {
            return headB;
        }
        if (headB==nullptr && headA != nullptr)
        {
            return headA;
        }
        if (headA==nullptr && headB==nullptr)
        {
            return nullptr;
        }
        auto ha = std::unordered_set<ListNode*>();
        auto hb = std::unordered_set<ListNode*>();
        auto cpa = headA;
        auto cpb = headB;
        int i = 1;
        while(true){
            if (cpa == nullptr && cpb == nullptr)
            {
                break;
            }
            if(i%2 ==0){
                if (cpa == nullptr && cpb != nullptr)
                {
                    i +=1;
                    continue;
                }
                if (hb.find(cpa) != hb.end())
                {
                    return cpa;
                }
                ha.insert(cpa);
                cpa = cpa->next;
            }
            if(i%2 != 0){
                if (cpb==nullptr && cpa != nullptr)
                {
                    i+=1;
                    continue;
                }
                if (ha.find(cpb) != ha.end())
                {
                    return cpb;
                }
                
                hb.insert(cpb);
                cpb = cpb->next;
            }
            i+=1;
        }
        return nullptr;
    }
    int majorityElement(vector<int>& nums) {
        int N = nums.size();
        auto h = std::unordered_map<int,int>();
        for (size_t i = 0; i < N; i++)
        {
            h.insert({nums[i],0});
        }
        for (size_t i = 0; i < N; i++)
        {
            h[nums[i]] += 1;
        }
        for(auto it: h){
            if(it.second > N/2){
                return it.first;
            }
        }

        
    }
    bool isHappy(int n) {
        
    }

};


int main(){
    auto l0 = ListNode(1);
    auto l1 = ListNode(2);
    auto l2 = ListNode(3);
    auto l3 = ListNode(1);
    l0.next = &l1;
    l1.next = &l2;
    l2.next = &l3;
    auto sol = Solution();
    std::cout << sol.hasCycle(&l0);
    return 0;
}

